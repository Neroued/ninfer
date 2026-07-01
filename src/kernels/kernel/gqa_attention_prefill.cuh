#pragma once

// qus::kernels - gqa_attention prefill kernels. The op writes the full prompt
// k/v into KVCache positions [0..T-1], then computes causal GQA attention for
// every prompt token with fp32 online softmax and fp32 AV accumulation.

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace qus::kernels {

inline constexpr int kGqaPrefillHeadDim   = 256;
inline constexpr int kGqaPrefillQHeads    = 24;
inline constexpr int kGqaPrefillKVHeads   = 4;
inline constexpr int kGqaPrefillGroupSize = 6;

__device__ __forceinline__ std::int64_t gqa_prefill_cache_index(int kv_head, int d, int position,
                                                                int padded_context) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kGqaPrefillHeadDim) *
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
    value = fmaxf(value, __shfl_down_sync(mask, value, 16));
    value = fmaxf(value, __shfl_down_sync(mask, value, 8));
    value = fmaxf(value, __shfl_down_sync(mask, value, 4));
    value = fmaxf(value, __shfl_down_sync(mask, value, 2));
    value = fmaxf(value, __shfl_down_sync(mask, value, 1));
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

__device__ __forceinline__ unsigned gqa_prefill_smem_addr(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void gqa_prefill_ldmatrix_x4(unsigned& a0, unsigned& a1, unsigned& a2,
                                                        unsigned& a3, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                 : "r"(addr));
}

__device__ __forceinline__ void gqa_prefill_ldmatrix_x2(unsigned& b0, unsigned& b1,
                                                        unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(b0), "=r"(b1)
                 : "r"(addr));
}

__device__ __forceinline__ void gqa_prefill_mma_m16n8k16_bf16(
    float& c0, float& c1, float& c2, float& c3, unsigned a0, unsigned a1, unsigned a2,
    unsigned a3, unsigned b0, unsigned b1) {
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
                                                  __nv_bfloat16* cache_k, __nv_bfloat16* cache_v,
                                                  std::int32_t tokens,
                                                  std::int32_t padded_context) {
    const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t n =
        static_cast<std::int64_t>(tokens) * kGqaPrefillKVHeads * kGqaPrefillHeadDim;
    if (idx >= n) { return; }

    const int d       = static_cast<int>(idx % kGqaPrefillHeadDim);
    const int tmp     = static_cast<int>(idx / kGqaPrefillHeadDim);
    const int kv_head = tmp % kGqaPrefillKVHeads;
    const int token   = tmp / kGqaPrefillKVHeads;

    const std::int64_t cache_off = gqa_prefill_cache_index(kv_head, d, token, padded_context);
    cache_k[cache_off]           = k[idx];
    cache_v[cache_off]           = v[idx];
}

__launch_bounds__(256) __global__
    void gqa_attention_prefill_slow_kernel(const __nv_bfloat16* q, const __nv_bfloat16* cache_k,
                                           const __nv_bfloat16* cache_v, float scale,
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

    float max_score = -3.4028234663852886e38f;
    float denom     = 0.0f;
    float acc       = 0.0f;
    for (std::int32_t j = 0; j <= token; ++j) {
        const float k_d =
            __bfloat162float(cache_k[gqa_prefill_cache_index(kv_head, d, j, padded_context)]);
        const float dot_part = q_d * k_d;
        const float score    = gqa_prefill_block_sum_256(dot_part, scratch) * scale;
        const float next_max = fmaxf(max_score, score);
        const float old_w    = expf(max_score - next_max);
        const float new_w    = expf(score - next_max);
        const float v_d =
            __bfloat162float(cache_v[gqa_prefill_cache_index(kv_head, d, j, padded_context)]);
        acc                  = acc * old_w + new_w * v_d;
        denom                = denom * old_w + new_w;
        max_score            = next_max;
    }

    out[gqa_prefill_q_index(q_head, d, token)] = __float2bfloat16(acc / denom);
}

__launch_bounds__(512, 1) __global__
    void gqa_attention_prefill_kernel(const __nv_bfloat16* q, const __nv_bfloat16* cache_k,
                                      const __nv_bfloat16* cache_v, float scale, __nv_bfloat16* out,
                                      std::int32_t tokens, std::int32_t padded_context) {
    constexpr int Br       = 32;
    constexpr int Bc       = 32;
    constexpr int D        = kGqaPrefillHeadDim;
    constexpr int Warps    = 16;
    constexpr int Threads  = Warps * 32;
    constexpr float Log2E  = 1.4426950408889634074f;

    static_assert(Threads == 512);

    __shared__ __align__(16) __nv_bfloat16 q_s[Br * D];
    __shared__ __align__(16) __nv_bfloat16 k_s[Bc * D];
    __shared__ __align__(16) __nv_bfloat16 vt_s[D * Bc];
    __shared__ __align__(16) __nv_bfloat16 p_s[Br * Bc];
    __shared__ __align__(16) float s_s[Br * Bc];
    __shared__ float row_m[Br];
    __shared__ float row_l[Br];
    __shared__ float row_alpha[Br];

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int q0      = q_block * Br;
    const int kv_head = q_head / kGqaPrefillGroupSize;

    if (q_head >= kGqaPrefillQHeads || q0 >= tokens) { return; }

    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    if (tid < Br) {
        row_m[tid]     = -CUDART_INF_F;
        row_l[tid]     = 0.0f;
        row_alpha[tid] = 0.0f;
    }

    for (int idx = tid; idx < Br * D; idx += Threads) {
        const int row  = idx / D;
        const int d    = idx - row * D;
        const int qrow = q0 + row;
        __nv_bfloat16 value = __float2bfloat16(0.0f);
        if (qrow < tokens) { value = q[gqa_prefill_q_index(q_head, d, qrow)]; }
        q_s[row * D + gqa_prefill_swz(row, d)] = value;
    }
    __syncthreads();

    float acc[4][4];
#pragma unroll
    for (int n = 0; n < 4; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }

    const int max_key = min(tokens, q0 + Br);
    for (int k0 = 0; k0 < max_key; k0 += Bc) {
        for (int idx = tid; idx < Bc * D; idx += Threads) {
            const int key_l = idx / D;
            const int d     = idx - key_l * D;
            const int key   = k0 + key_l;
            __nv_bfloat16 kval = __float2bfloat16(0.0f);
            __nv_bfloat16 vval = __float2bfloat16(0.0f);
            if (key < tokens) {
                const std::int64_t off = gqa_prefill_cache_index(kv_head, d, key, padded_context);
                kval                   = cache_k[off];
                vval                   = cache_v[off];
            }
            k_s[key_l * D + gqa_prefill_swz(key_l, d)] = kval;
            vt_s[d * Bc + gqa_prefill_swz32(d, key_l)] = vval;
        }
        __syncthreads();

        if (warp < 8) {
            float score[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const int row_base = (warp >> 2) * 16;
            const int n_tile   = warp & 3;
#pragma unroll
            for (int ks = 0; ks < D; ks += 16) {
                unsigned af[4];
                unsigned bf[2];
                const int arow = row_base + a_rowoff;
                const int acol = ks + a_coloff;
                const int brow = n_tile * 8 + b_rin;
                const int bcol = ks + b_koff;
                gqa_prefill_ldmatrix_x4(af[0], af[1], af[2], af[3],
                                        gqa_prefill_smem_addr(
                                            &q_s[arow * D + gqa_prefill_swz(arow, acol)]));
                gqa_prefill_ldmatrix_x2(bf[0], bf[1],
                                        gqa_prefill_smem_addr(
                                            &k_s[brow * D + gqa_prefill_swz(brow, bcol)]));
                gqa_prefill_mma_m16n8k16_bf16(score[0], score[1], score[2], score[3], af[0],
                                              af[1], af[2], af[3], bf[0], bf[1]);
            }

            const int col0 = n_tile * 8 + 2 * lid;
            const int col1 = col0 + 1;
            const int row0 = row_base + gid;
            const int row1 = row_base + gid + 8;
            const int key0 = k0 + col0;
            const int key1 = k0 + col1;
            const int qrow0 = q0 + row0;
            const int qrow1 = q0 + row1;

            s_s[row0 * Bc + col0] =
                (qrow0 < tokens && key0 < tokens && key0 <= qrow0) ? score[0] * scale
                                                                    : -CUDART_INF_F;
            s_s[row0 * Bc + col1] =
                (qrow0 < tokens && key1 < tokens && key1 <= qrow0) ? score[1] * scale
                                                                    : -CUDART_INF_F;
            s_s[row1 * Bc + col0] =
                (qrow1 < tokens && key0 < tokens && key0 <= qrow1) ? score[2] * scale
                                                                    : -CUDART_INF_F;
            s_s[row1 * Bc + col1] =
                (qrow1 < tokens && key1 < tokens && key1 <= qrow1) ? score[3] * scale
                                                                    : -CUDART_INF_F;
        }
        __syncthreads();

        if (warp < 16) {
            constexpr unsigned FullMask = 0xffffffffu;
#pragma unroll
            for (int row_iter = 0; row_iter < 2; ++row_iter) {
                const int row  = warp + row_iter * 16;
                const int qrow = q0 + row;
                const float s  = s_s[row * Bc + lane];
                float block_m  = gqa_prefill_warp_max(s);
                block_m        = __shfl_sync(FullMask, block_m, 0);

                const float old_m = row_m[row];
                const float new_m = fmaxf(old_m, block_m);
                float alpha       = 0.0f;
                if (old_m != -CUDART_INF_F && new_m != -CUDART_INF_F) {
                    alpha = gqa_prefill_exp2_fast((old_m - new_m) * Log2E);
                }

                float p = 0.0f;
                if (qrow < tokens && s != -CUDART_INF_F) {
                    p = gqa_prefill_exp2_fast((s - new_m) * Log2E);
                }
                float block_l = gqa_prefill_warp_sum(p);
                block_l       = __shfl_sync(FullMask, block_l, 0);
                p_s[row * Bc + gqa_prefill_swz32(row, lane)] = __float2bfloat16(p);

                if (lane == 0) {
                    row_m[row]     = new_m;
                    row_l[row]     = row_l[row] * alpha + block_l;
                    row_alpha[row] = alpha;
                }
            }
        }
        __syncthreads();

#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int row_base = (warp >> 3) * 16;
            const float a0 = row_alpha[row_base + gid];
            const float a1 = row_alpha[row_base + gid + 8];
            acc[n][0] *= a0;
            acc[n][1] *= a0;
            acc[n][2] *= a1;
            acc[n][3] *= a1;
        }

        const int row_base = (warp >> 3) * 16;
        const int d_base   = (warp & 7) * 32;
#pragma unroll
        for (int ks = 0; ks < Bc; ks += 16) {
            unsigned pf[4];
            const int prow = row_base + a_rowoff;
            const int pcol = ks + a_coloff;
            gqa_prefill_ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                                    gqa_prefill_smem_addr(
                                        &p_s[prow * Bc + gqa_prefill_swz32(prow, pcol)]));
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                unsigned vf[2];
                const int vrow = d_base + n * 8 + b_rin;
                const int vcol = ks + b_koff;
                gqa_prefill_ldmatrix_x2(vf[0], vf[1],
                                        gqa_prefill_smem_addr(
                                            &vt_s[vrow * Bc + gqa_prefill_swz32(vrow, vcol)]));
                gqa_prefill_mma_m16n8k16_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3],
                                              pf[0], pf[1], pf[2], pf[3], vf[0], vf[1]);
            }
        }
        __syncthreads();
    }

    const int row_base = (warp >> 3) * 16;
    const int d_base   = (warp & 7) * 32;
#pragma unroll
    for (int n = 0; n < 4; ++n) {
        const int d0 = d_base + n * 8 + 2 * lid;
        const int d1 = d0 + 1;
        const int row0 = row_base + gid;
        const int row1 = row_base + gid + 8;
        const int qrow0 = q0 + row0;
        const int qrow1 = q0 + row1;
        const float l0 = row_l[row0];
        const float l1 = row_l[row1];
        if (qrow0 < tokens) {
            if (d0 < D) {
                out[gqa_prefill_q_index(q_head, d0, qrow0)] =
                    __float2bfloat16((l0 > 0.0f) ? (acc[n][0] / l0) : 0.0f);
            }
            if (d1 < D) {
                out[gqa_prefill_q_index(q_head, d1, qrow0)] =
                    __float2bfloat16((l0 > 0.0f) ? (acc[n][1] / l0) : 0.0f);
            }
        }
        if (qrow1 < tokens) {
            if (d0 < D) {
                out[gqa_prefill_q_index(q_head, d0, qrow1)] =
                    __float2bfloat16((l1 > 0.0f) ? (acc[n][2] / l1) : 0.0f);
            }
            if (d1 < D) {
                out[gqa_prefill_q_index(q_head, d1, qrow1)] =
                    __float2bfloat16((l1 > 0.0f) ? (acc[n][3] / l1) : 0.0f);
            }
        }
    }
}

} // namespace qus::kernels
