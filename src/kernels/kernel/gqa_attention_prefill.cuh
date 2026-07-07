#pragma once

// qus::kernels - gqa_attention prompt-scale kernels. The op writes k/v into
// absolute KVCache positions from the device positions vector, then computes
// causal GQA attention for every chunk token over all cached history.

#include <cuda_bf16.h>
#include <math_constants.h>

#include "kernels/kernel/gdn_common.cuh"

#include <cstdint>

namespace qus::kernels {

inline constexpr int kGqaPrefillHeadDim   = 256;
inline constexpr int kGqaPrefillQHeads    = 24;
inline constexpr int kGqaPrefillKVHeads   = 4;
inline constexpr int kGqaPrefillGroupSize = 6;

__device__ __forceinline__ std::int64_t gqa_prefill_cache_index(int kv_head, int d, int position,
                                                                int padded_context) {
    return static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kGqaPrefillHeadDim) *
                                              (static_cast<std::int64_t>(position) +
                                               static_cast<std::int64_t>(padded_context) * kv_head);
}

__device__ __forceinline__ std::int64_t gqa_prefill_q_index(int q_head, int d, int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kGqaPrefillHeadDim) *
               (static_cast<std::int64_t>(q_head) +
                static_cast<std::int64_t>(kGqaPrefillQHeads) * token);
}

__device__ __forceinline__ std::int64_t gqa_prefill_kv_index(int kv_head, int d, int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kGqaPrefillHeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(kGqaPrefillKVHeads) * token);
}

__device__ __forceinline__ float gqa_prefill_warp_sum(float value) {
    constexpr unsigned mask = 0xffffffffu;
    value += __shfl_down_sync(mask, value, 16);
    value += __shfl_down_sync(mask, value, 8);
    value += __shfl_down_sync(mask, value, 4);
    value += __shfl_down_sync(mask, value, 2);
    value += __shfl_down_sync(mask, value, 1);
    return value;
}

__device__ __forceinline__ float gqa_prefill_warp_max(float value) {
    constexpr unsigned mask = 0xffffffffu;
    value                   = fmaxf(value, __shfl_down_sync(mask, value, 16));
    value                   = fmaxf(value, __shfl_down_sync(mask, value, 8));
    value                   = fmaxf(value, __shfl_down_sync(mask, value, 4));
    value                   = fmaxf(value, __shfl_down_sync(mask, value, 2));
    value                   = fmaxf(value, __shfl_down_sync(mask, value, 1));
    return value;
}

__device__ __forceinline__ float gqa_prefill_block_sum_256(float value, float* scratch) {
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    value = gqa_prefill_warp_sum(value);
    if (lane == 0) { scratch[warp] = value; }
    __syncthreads();

    value = (tid < 8) ? scratch[lane] : 0.0f;
    if (warp == 0) { value = gqa_prefill_warp_sum(value); }
    if (tid == 0) { scratch[0] = value; }
    __syncthreads();
    value = scratch[0];
    __syncthreads();
    return value;
}

__device__ __forceinline__ int gqa_prefill_swz(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

__device__ __forceinline__ int gqa_prefill_swz32(int row, int col) {
    return (((col >> 3) ^ (row & 3)) << 3) | (col & 7);
}

__device__ __forceinline__ unsigned gqa_prefill_pack_bf16(float lo, float hi) {
    const __nv_bfloat16 lo_b = __float2bfloat16(lo);
    const __nv_bfloat16 hi_b = __float2bfloat16(hi);
    const unsigned lo_bits   = static_cast<__nv_bfloat16_raw>(lo_b).x;
    const unsigned hi_bits   = static_cast<__nv_bfloat16_raw>(hi_b).x;
    return lo_bits | (hi_bits << 16);
}

__device__ __forceinline__ unsigned gqa_prefill_smem_addr(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void gqa_prefill_ldmatrix_x4(unsigned& a0, unsigned& a1, unsigned& a2,
                                                        unsigned& a3, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                 : "r"(addr));
}

__device__ __forceinline__ void gqa_prefill_ldmatrix_x2(unsigned& b0, unsigned& b1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(b0), "=r"(b1)
                 : "r"(addr));
}

__device__ __forceinline__ void gqa_prefill_ldmatrix_x2_trans(unsigned& b0, unsigned& b1,
                                                              unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(b0), "=r"(b1)
                 : "r"(addr));
}

__device__ __forceinline__ void gqa_prefill_mma_m16n8k16_bf16(float& c0, float& c1, float& c2,
                                                              float& c3, unsigned a0, unsigned a1,
                                                              unsigned a2, unsigned a3, unsigned b0,
                                                              unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float gqa_prefill_exp2_fast(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__global__ void gqa_attention_prefill_fill_kernel(const __nv_bfloat16* k, const __nv_bfloat16* v,
                                                  const std::int32_t* positions,
                                                  __nv_bfloat16* cache_k, __nv_bfloat16* cache_v,
                                                  std::int32_t tokens,
                                                  std::int32_t padded_context) {
    const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t n =
        static_cast<std::int64_t>(tokens) * kGqaPrefillKVHeads * kGqaPrefillHeadDim;
    if (idx >= n) { return; }

    const int d        = static_cast<int>(idx % kGqaPrefillHeadDim);
    const int tmp      = static_cast<int>(idx / kGqaPrefillHeadDim);
    const int kv_head  = tmp % kGqaPrefillKVHeads;
    const int token    = tmp / kGqaPrefillKVHeads;
    const int position = positions[0] + token;

    const std::int64_t cache_off = gqa_prefill_cache_index(kv_head, d, position, padded_context);
    cache_k[cache_off]           = k[idx];
    cache_v[cache_off]           = v[idx];
}

__launch_bounds__(256) __global__
    void gqa_attention_prefill_slow_kernel(const __nv_bfloat16* q, const __nv_bfloat16* cache_k,
                                           const __nv_bfloat16* cache_v,
                                           const std::int32_t* positions, float scale,
                                           __nv_bfloat16* out, std::int32_t tokens,
                                           std::int32_t padded_context) {
    const int block  = static_cast<int>(blockIdx.x);
    const int q_head = block % kGqaPrefillQHeads;
    const int token  = block / kGqaPrefillQHeads;
    const int d      = threadIdx.x;
    if (q_head >= kGqaPrefillQHeads || token >= tokens || d >= kGqaPrefillHeadDim) { return; }

    __shared__ float scratch[kGqaPrefillHeadDim];

    const int kv_head = q_head / kGqaPrefillGroupSize;
    const float q_d   = __bfloat162float(q[gqa_prefill_q_index(q_head, d, token)]);

    float max_score              = -3.4028234663852886e38f;
    float denom                  = 0.0f;
    float acc                    = 0.0f;
    const std::int32_t query_abs = positions[token];
    for (std::int32_t j = 0; j <= query_abs; ++j) {
        const float k_d =
            __bfloat162float(cache_k[gqa_prefill_cache_index(kv_head, d, j, padded_context)]);
        const float dot_part = q_d * k_d;
        const float score    = gqa_prefill_block_sum_256(dot_part, scratch) * scale;
        const float next_max = fmaxf(max_score, score);
        const float old_w    = expf(max_score - next_max);
        const float new_w    = expf(score - next_max);
        const float v_d =
            __bfloat162float(cache_v[gqa_prefill_cache_index(kv_head, d, j, padded_context)]);
        acc       = acc * old_w + new_w * v_d;
        denom     = denom * old_w + new_w;
        max_score = next_max;
    }

    out[gqa_prefill_q_index(q_head, d, token)] = __float2bfloat16(acc / denom);
}

// Per-warp-independent flash prefill: each warp owns 16 query rows and runs its own
// QK -> register online softmax -> PV pipeline. The only CTA barrier is the one that
// publishes the shared K/V tile per key block; there are no full-CTA barriers between the
// compute phases, so the warps' phases interleave and keep the tensor pipe fed at low
// occupancy (the head_dim=256 O accumulator is register-heavy, so occupancy is low by design
// and latency is hidden by the per-warp pipeline, mirroring the linear tensor-core GEMM).
__launch_bounds__(128, 2) __global__
    void gqa_attention_prefill_kernel(const __nv_bfloat16* q, const __nv_bfloat16* cache_k,
                                      const __nv_bfloat16* cache_v, const std::int32_t* positions,
                                      float scale, __nv_bfloat16* out, std::int32_t tokens,
                                      std::int32_t padded_context) {
    constexpr int Wc            = 4;
    constexpr int Br            = Wc * 16;
    constexpr int Bc            = 32;
    constexpr int D             = kGqaPrefillHeadDim;
    constexpr int Threads       = Wc * 32;
    constexpr int QKNt          = Bc / 8;  // QK score n-tiles
    constexpr int QKKs          = D / 16;  // QK contraction steps over head_dim
    constexpr int PVNt          = D / 8;   // PV output n-tiles
    constexpr int PVKs          = Bc / 16; // PV contraction steps over keys
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(Threads == 128);
    static_assert(Br * D == 2 * Bc * D, "Q-load tile must alias the K/V staging tile");

    // Q is staged here, copied into registers, then this buffer is reused for the K/V tiles
    // (Br*D == 2*Bc*D). P stays in registers and feeds the PV MMA directly.
    __shared__ __align__(16) __nv_bfloat16 qkv_s[Br * D];
    __nv_bfloat16* k_s = qkv_s;
    __nv_bfloat16* v_s = qkv_s + Bc * D;

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int q0      = q_block * Br;
    const int kv_head = q_head / kGqaPrefillGroupSize;
    const int base_pos = positions[0];

    if (q_head >= kGqaPrefillQHeads || q0 >= tokens) { return; }

    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    const int warp_row0 = warp * 16;            // CTA-tile rows owned by this warp

    for (int idx = tid; idx < Br * D; idx += Threads) {
        const int row       = idx / D;
        const int d         = idx - row * D;
        const int qrow      = q0 + row;
        __nv_bfloat16 value = __float2bfloat16(0.0f);
        if (qrow < tokens) { value = q[gqa_prefill_q_index(q_head, d, qrow)]; }
        qkv_s[row * D + gqa_prefill_swz(row, d)] = value;
    }
    __syncthreads();

    // Copy this warp's Q rows into registers (all QK contraction steps), then release the
    // staging buffer so the main loop can reuse it for K/V.
    unsigned af_q[QKKs][4];
#pragma unroll
    for (int k = 0; k < QKKs; ++k) {
        const int arow = warp_row0 + a_rowoff;
        const int acol = k * 16 + a_coloff;
        gqa_prefill_ldmatrix_x4(
            af_q[k][0], af_q[k][1], af_q[k][2], af_q[k][3],
            gqa_prefill_smem_addr(&qkv_s[arow * D + gqa_prefill_swz(arow, acol)]));
    }
    __syncthreads();

    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;

    const int tile_rows     = min(Br, tokens - q0);
    const int last_qrow     = q0 + tile_rows - 1;
    const int max_query_abs = (tile_rows > 0) ? base_pos + last_qrow : -1;
    const int key_blocks    = (max_query_abs + Bc) / Bc;

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = kb * Bc;
        const bool full_kv_tile = (k0 + Bc - 1) <= max_query_abs;
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 8); chunk += Threads) {
            const int key_l      = chunk / (D / 8);
            const int d          = (chunk - key_l * (D / 8)) * 8;
            const int key        = k0 + key_l;
            __nv_bfloat16* k_dst = &k_s[key_l * D + gqa_prefill_swz(key_l, d)];
            __nv_bfloat16* v_dst = &v_s[key_l * D + gqa_prefill_swz(key_l, d)];
            if (full_kv_tile || key <= max_query_abs) {
                const std::int64_t off = gqa_prefill_cache_index(kv_head, d, key, padded_context);
                qus::kernels::async_copy_global_to_shared<16>(k_dst, &cache_k[off]);
                qus::kernels::async_copy_global_to_shared<16>(v_dst, &cache_v[off]);
            } else {
                *reinterpret_cast<int4*>(k_dst) = make_int4(0, 0, 0, 0);
                *reinterpret_cast<int4*>(v_dst) = make_int4(0, 0, 0, 0);
            }
        }
        qus::kernels::async_copy_commit();
        qus::kernels::async_copy_wait<0>();
        __syncthreads();

        // QK: this warp computes S[16 rows][Bc] entirely in registers.
        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
        }
#pragma unroll
        for (int k = 0; k < QKKs; ++k) {
            unsigned bf[QKNt][2];
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int brow = nt * 8 + b_rin;
                const int bcol = k * 16 + b_koff;
                gqa_prefill_ldmatrix_x2(
                    bf[nt][0], bf[nt][1],
                    gqa_prefill_smem_addr(&k_s[brow * D + gqa_prefill_swz(brow, bcol)]));
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                gqa_prefill_mma_m16n8k16_bf16(score[nt][0], score[nt][1], score[nt][2],
                                              score[nt][3], af_q[k][0], af_q[k][1], af_q[k][2],
                                              af_q[k][3], bf[nt][0], bf[nt][1]);
            }
        }

        const int row0  = warp_row0 + gid;
        const int row1  = warp_row0 + gid + 8;
        const int qrow0 = q0 + row0;
        const int qrow1 = q0 + row1;
        const int qabs0 = (qrow0 < tokens) ? base_pos + qrow0 : -1;
        const int qabs1 = (qrow1 < tokens) ? base_pos + qrow1 : -1;
        const bool full_score_tile = (q0 + Br <= tokens) && ((k0 + Bc - 1) <= (base_pos + q0));

        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
        if (full_score_tile) {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                score[nt][0] *= scale;
                score[nt][1] *= scale;
                score[nt][2] *= scale;
                score[nt][3] *= scale;
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int col0 = nt * 8 + 2 * lid;
                const int col1 = col0 + 1;
                const int key0 = k0 + col0;
                const int key1 = k0 + col1;
                score[nt][0]   = (qrow0 < tokens && key0 <= max_query_abs && key0 <= qabs0)
                                     ? score[nt][0] * scale
                                     : -CUDART_INF_F;
                score[nt][1]   = (qrow0 < tokens && key1 <= max_query_abs && key1 <= qabs0)
                                     ? score[nt][1] * scale
                                     : -CUDART_INF_F;
                score[nt][2]   = (qrow1 < tokens && key0 <= max_query_abs && key0 <= qabs1)
                                     ? score[nt][2] * scale
                                     : -CUDART_INF_F;
                score[nt][3]   = (qrow1 < tokens && key1 <= max_query_abs && key1 <= qabs1)
                                     ? score[nt][3] * scale
                                     : -CUDART_INF_F;
                bm0            = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1            = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        }
        bm0 = fmaxf(bm0, __shfl_xor_sync(FullMask, bm0, 1));
        bm0 = fmaxf(bm0, __shfl_xor_sync(FullMask, bm0, 2));
        bm1 = fmaxf(bm1, __shfl_xor_sync(FullMask, bm1, 1));
        bm1 = fmaxf(bm1, __shfl_xor_sync(FullMask, bm1, 2));

        const float nm0 = fmaxf(m0, bm0);
        const float nm1 = fmaxf(m1, bm1);
        const float alpha0 =
            (m0 == -CUDART_INF_F) ? 0.0f : gqa_prefill_exp2_fast((m0 - nm0) * Log2E);
        const float alpha1 =
            (m1 == -CUDART_INF_F) ? 0.0f : gqa_prefill_exp2_fast((m1 - nm1) * Log2E);

        float bl0 = 0.0f, bl1 = 0.0f;
        unsigned p_frag[PVKs][4];
        if (full_score_tile) {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const float p00 = gqa_prefill_exp2_fast((score[nt][0] - nm0) * Log2E);
                const float p01 = gqa_prefill_exp2_fast((score[nt][1] - nm0) * Log2E);
                const float p10 = gqa_prefill_exp2_fast((score[nt][2] - nm1) * Log2E);
                const float p11 = gqa_prefill_exp2_fast((score[nt][3] - nm1) * Log2E);
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                const int pk = nt >> 1;
                if ((nt & 1) == 0) {
                    p_frag[pk][0] = gqa_prefill_pack_bf16(p00, p01);
                    p_frag[pk][1] = gqa_prefill_pack_bf16(p10, p11);
                } else {
                    p_frag[pk][2] = gqa_prefill_pack_bf16(p00, p01);
                    p_frag[pk][3] = gqa_prefill_pack_bf16(p10, p11);
                }
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const float p00 = (nm0 > -CUDART_INF_F && score[nt][0] > -CUDART_INF_F)
                                      ? gqa_prefill_exp2_fast((score[nt][0] - nm0) * Log2E)
                                      : 0.0f;
                const float p01 = (nm0 > -CUDART_INF_F && score[nt][1] > -CUDART_INF_F)
                                      ? gqa_prefill_exp2_fast((score[nt][1] - nm0) * Log2E)
                                      : 0.0f;
                const float p10 = (nm1 > -CUDART_INF_F && score[nt][2] > -CUDART_INF_F)
                                      ? gqa_prefill_exp2_fast((score[nt][2] - nm1) * Log2E)
                                      : 0.0f;
                const float p11 = (nm1 > -CUDART_INF_F && score[nt][3] > -CUDART_INF_F)
                                      ? gqa_prefill_exp2_fast((score[nt][3] - nm1) * Log2E)
                                      : 0.0f;
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                const int pk = nt >> 1;
                if ((nt & 1) == 0) {
                    p_frag[pk][0] = gqa_prefill_pack_bf16(p00, p01);
                    p_frag[pk][1] = gqa_prefill_pack_bf16(p10, p11);
                } else {
                    p_frag[pk][2] = gqa_prefill_pack_bf16(p00, p01);
                    p_frag[pk][3] = gqa_prefill_pack_bf16(p10, p11);
                }
            }
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

        // PV: acc += P * V, contracting over the Bc keys. P is already in the
        // row-major A fragment layout that ldmatrix.x4 would have loaded from shared memory.
#pragma unroll
        for (int k = 0; k < PVKs; ++k) {
#pragma unroll
            for (int n = 0; n < PVNt; ++n) {
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = n * 8;
                gqa_prefill_ldmatrix_x2_trans(
                    vf[0], vf[1],
                    gqa_prefill_smem_addr(&v_s[vrow * D + gqa_prefill_swz(vrow, vcol)]));
                gqa_prefill_mma_m16n8k16_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3],
                                              p_frag[k][0], p_frag[k][1], p_frag[k][2],
                                              p_frag[k][3], vf[0], vf[1]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
        const int d0    = n * 8 + 2 * lid;
        const int d1    = d0 + 1;
        const int qrow0 = q0 + warp_row0 + gid;
        const int qrow1 = q0 + warp_row0 + gid + 8;
        if (qrow0 < tokens) {
            out[gqa_prefill_q_index(q_head, d0, qrow0)] =
                __float2bfloat16((l0 > 0.0f) ? (acc[n][0] / l0) : 0.0f);
            out[gqa_prefill_q_index(q_head, d1, qrow0)] =
                __float2bfloat16((l0 > 0.0f) ? (acc[n][1] / l0) : 0.0f);
        }
        if (qrow1 < tokens) {
            out[gqa_prefill_q_index(q_head, d0, qrow1)] =
                __float2bfloat16((l1 > 0.0f) ? (acc[n][2] / l1) : 0.0f);
            out[gqa_prefill_q_index(q_head, d1, qrow1)] =
                __float2bfloat16((l1 > 0.0f) ? (acc[n][3] / l1) : 0.0f);
        }
    }
}

} // namespace qus::kernels
