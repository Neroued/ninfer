#pragma once

// qus::kernels - gqa_attention prompt-scale kernels, rewritten to mirror the
// FlashAttention-2 forward kernel (Flash_fwd_kernel_traits<256, 64, 64, 4>) that
// flash-attn selects for our shape on the RTX 5090 (bf16, causal, head_dim 256):
//
//   * Br = 64 query rows and Bc = 64 key columns per CTA tile.
//   * 4 warps / 128 threads; each warp owns 16 query rows of the tile.
//   * Q, K, V staged in 96 KiB of dynamic shared memory (single-buffered), with
//     the cp.async of the next K/V tile overlapped against the current
//     QK / PV tensor-core work (exactly FA's single-buffer overlap pattern).
//   * m16n8k16 bf16 MMA for both S = Q Kᵀ and O += P V, online softmax in exp2.
//
// The op still writes the new chunk k/v into absolute KVCache positions first
// (gqa_attention_prefill_fill_kernel), then computes causal GQA attention for
// every chunk token over all cached history using bottom-right causal alignment
// (query row i attends to keys [0, base_pos + i]).

#include <cuda_bf16.h>
#include <math_constants.h>

#include "kernels/kernel/gdn_common.cuh"
#include "kernels/kernel/gqa_attention_kv_quant.cuh"

#include <cstdint>

namespace qus::kernels {

inline constexpr int kGqaPrefillHeadDim   = 256;
inline constexpr int kGqaPrefillQHeads    = 24;
inline constexpr int kGqaPrefillKVHeads   = 4;
inline constexpr int kGqaPrefillGroupSize = 6;

// FA tile geometry for this shape.
inline constexpr int kGqaPrefillBr        = 64;
inline constexpr int kGqaPrefillBc        = 64;
inline constexpr int kGqaPrefillThreads   = 128;
inline constexpr int kGqaPrefillSmemBytes = (kGqaPrefillBr + 2 * kGqaPrefillBc) *
                                            kGqaPrefillHeadDim *
                                            static_cast<int>(sizeof(__nv_bfloat16));

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

// 128-bit (8 bf16) XOR swizzle over an 8-row period, matching the CUTLASS
// Swizzle<3,3,3> layout FA uses for its Q/K/V smem tiles. Conflict-free for the
// ldmatrix accesses below.
__device__ __forceinline__ int gqa_prefill_swz(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

__device__ __forceinline__ unsigned gqa_prefill_pack_bf16(float lo, float hi) {
    unsigned out;
    const unsigned lo_bits = __float_as_uint(lo);
    const unsigned hi_bits = __float_as_uint(hi);
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(out) : "r"(hi_bits), "r"(lo_bits));
    return out;
}

__device__ __forceinline__ unsigned gqa_prefill_smem_addr(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void gqa_prefill_cp_async_cg_16(void* smem_dst, const void* gmem_src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(
                     static_cast<unsigned>(__cvta_generic_to_shared(smem_dst))),
                 "l"(gmem_src));
}

// Shared-memory byte address of a swizzled element from per-lane precomputed
// pieces. For (row, col) with col a multiple of 8 the swizzled byte offset is
// 512*row + (((col>>3)<<4) ^ ((row&7)<<4)); we fold 512*row (plus the buffer
// base) into lane_base, (row&7)<<4 into r, and split (col>>3)<<4 into a
// compile-time immediate ck plus the per-lane low bit as. Each ldmatrix address
// is then a single LOP3 + IADD3.
__device__ __forceinline__ unsigned gqa_prefill_swz_addr(unsigned lane_base, unsigned ck,
                                                         unsigned as, unsigned r) {
    return lane_base + ((ck | as) ^ r);
}

__device__ __forceinline__ void gqa_prefill_ldmatrix_x4(unsigned& a0, unsigned& a1, unsigned& a2,
                                                        unsigned& a3, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                 : "r"(addr));
}

__device__ __forceinline__ void gqa_prefill_ldmatrix_x4_trans(unsigned& b0, unsigned& b1,
                                                              unsigned& b2, unsigned& b3,
                                                              unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3)
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

__global__ void gqa_attention_prefill_fill_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, __nv_bfloat16* __restrict__ cache_k,
    __nv_bfloat16* __restrict__ cache_v, std::int32_t tokens, std::int32_t padded_context) {
    constexpr int VecElems = 8; // 8 bf16 == 16 B, matching the cache row alignment.
    const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t n =
        static_cast<std::int64_t>(tokens) * kGqaPrefillKVHeads * (kGqaPrefillHeadDim / VecElems);
    if (idx >= n) { return; }

    const int vec      = static_cast<int>(idx % (kGqaPrefillHeadDim / VecElems));
    const int tmp      = static_cast<int>(idx / (kGqaPrefillHeadDim / VecElems));
    const int kv_head  = tmp % kGqaPrefillKVHeads;
    const int token    = tmp / kGqaPrefillKVHeads;
    const int d        = vec * VecElems;
    const int position = positions[0] + token;

    const std::int64_t src_off =
        static_cast<std::int64_t>(d) +
        static_cast<std::int64_t>(kGqaPrefillHeadDim) * (kv_head + kGqaPrefillKVHeads * token);
    const std::int64_t cache_off = gqa_prefill_cache_index(kv_head, d, position, padded_context);
    *reinterpret_cast<int4*>(&cache_k[cache_off]) = *reinterpret_cast<const int4*>(&k[src_off]);
    *reinterpret_cast<int4*>(&cache_v[cache_off]) = *reinterpret_cast<const int4*>(&v[src_off]);
}

// int8 counterpart of the prefill fill: quantize the new chunk's K/V into the
// int8 cache (per 64-group absmax -> int8 code + fp16 scale). One CTA per
// (group, kv_head, token); the 64 threads own one 64-group and reduce its absmax
// in shared memory. This is the prefill fill for the quantized cache, not a
// standalone quant pass: the attention kernel below reads it straight back.
__global__ void gqa_attention_prefill_fill_i8_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, std::int8_t* __restrict__ cache_k,
    std::int8_t* __restrict__ cache_v, __half* __restrict__ scale_k, __half* __restrict__ scale_v,
    std::int32_t tokens, std::int32_t padded_context) {
    __shared__ float k_abs[kGqaKvQuantGroup];
    __shared__ float v_abs[kGqaKvQuantGroup];

    const int group   = static_cast<int>(blockIdx.x);
    const int kv_head = static_cast<int>(blockIdx.y);
    const int token   = static_cast<int>(blockIdx.z);
    const int lane    = static_cast<int>(threadIdx.x);
    if (group >= kGqaKvQuantGroups || kv_head >= kGqaPrefillKVHeads || token >= tokens ||
        lane >= kGqaKvQuantGroup) {
        return;
    }

    const int position         = positions[0] + token;
    const int d                = group * kGqaKvQuantGroup + lane;
    const std::int64_t src_off = gqa_kv_quant_src_index(kv_head, d, token);
    const float k_value        = __bfloat162float(k[src_off]);
    const float v_value        = __bfloat162float(v[src_off]);
    k_abs[lane]                = fabsf(k_value);
    v_abs[lane]                = fabsf(v_value);
    __syncthreads();
    for (int stride = kGqaKvQuantGroup / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            k_abs[lane] = fmaxf(k_abs[lane], k_abs[lane + stride]);
            v_abs[lane] = fmaxf(v_abs[lane], v_abs[lane + stride]);
        }
        __syncthreads();
    }

    const __half ksh   = __float2half_rn(k_abs[0] > 0.0f ? k_abs[0] / 127.0f : 0.0f);
    const __half vsh   = __float2half_rn(v_abs[0] > 0.0f ? v_abs[0] / 127.0f : 0.0f);
    const float k_scl  = __half2float(ksh);
    const float v_scl  = __half2float(vsh);
    const float k_inv  = k_scl > 0.0f ? 1.0f / k_scl : 0.0f;
    const float v_inv  = v_scl > 0.0f ? 1.0f / v_scl : 0.0f;
    const std::int64_t code_off = gqa_kv_quant_code_index(kv_head, d, position, padded_context);
    cache_k[code_off]           = gqa_kv_quant_code(k_value, k_inv);
    cache_v[code_off]           = gqa_kv_quant_code(v_value, v_inv);
    if (lane == 0) {
        const std::int64_t scale_off =
            gqa_kv_quant_scale_index(kv_head, group, position, padded_context);
        scale_k[scale_off] = ksh;
        scale_v[scale_off] = vsh;
    }
}

// Stage one [Bc, D] K or V tile from the per-kv-head contiguous cache into the
// swizzled smem buffer. Keys beyond max_query_abs (which the causal mask always
// drops) are zeroed so the padded/uninitialized cache tail never feeds NaNs into
// the tensor cores. Mirrors FA's predicated K/V cp.async + Clear_OOB path.
__device__ __forceinline__ void gqa_prefill_stage_kv(__nv_bfloat16* dst, const __nv_bfloat16* cache,
                                                     int kv_head, int k0, int max_query_abs,
                                                     int padded_context, int tid) {
    constexpr int D         = kGqaPrefillHeadDim;
    constexpr int Bc        = kGqaPrefillBc;
    constexpr int Threads   = kGqaPrefillThreads;
    constexpr int VecPerRow = D / 8; // 8 bf16 per 16B cp.async
    const bool full_tile    = (k0 + Bc - 1) <= max_query_abs;
    // Block base pointer computed once (int64); per-element offsets stay 32-bit.
    const __nv_bfloat16* cache_block =
        cache + gqa_prefill_cache_index(kv_head, 0, k0, padded_context);
    if (full_tile) {
#pragma unroll
        for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
            const int key_l  = chunk >> 5;        // / VecPerRow (32)
            const int d      = (chunk & 31) << 3; // (chunk % 32) * 8
            __nv_bfloat16* p = &dst[key_l * D + gqa_prefill_swz(key_l, d)];
            gqa_prefill_cp_async_cg_16(p, &cache_block[key_l * D + d]);
        }
    } else {
#pragma unroll
        for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
            const int key_l  = chunk >> 5;        // / VecPerRow (32)
            const int d      = (chunk & 31) << 3; // (chunk % 32) * 8
            __nv_bfloat16* p = &dst[key_l * D + gqa_prefill_swz(key_l, d)];
            if ((k0 + key_l) <= max_query_abs) {
                gqa_prefill_cp_async_cg_16(p, &cache_block[key_l * D + d]);
            } else {
                *reinterpret_cast<int4*>(p) = make_int4(0, 0, 0, 0);
            }
        }
    }
}

__device__ __forceinline__ void
gqa_prefill_stage_kv_i8(__nv_bfloat16* dst, const std::int8_t* cache, const __half* scale,
                        int kv_head, int k0, int max_query_abs, int padded_context, int tid) {
    constexpr int D         = kGqaPrefillHeadDim;
    constexpr int Bc        = kGqaPrefillBc;
    constexpr int Threads   = kGqaPrefillThreads;
    constexpr int VecPerRow = D / 8;
#pragma unroll
    for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
        const int key_l  = chunk >> 5;
        const int d      = (chunk & 31) << 3;
        __nv_bfloat16* p = &dst[key_l * D + gqa_prefill_swz(key_l, d)];
        if ((k0 + key_l) <= max_query_abs) {
            *reinterpret_cast<int4*>(p) =
                gqa_kv_dequant_i8x8(cache, scale, kv_head, d, k0 + key_l, padded_context);
        } else {
            *reinterpret_cast<int4*>(p) = make_int4(0, 0, 0, 0);
        }
    }
}

// FlashAttention-2 forward, one CTA per (query 64-row block, query head). Grid is
// (ceil(tokens/64), q_heads). seqlen_q = tokens, seqlen_k = base_pos + tokens, with
// bottom-right causal alignment (query row i sees keys [0, base_pos + i]).
template <bool Quantized>
__launch_bounds__(kGqaPrefillThreads, 1) __global__ void gqa_attention_prefill_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ cache_k,
    const __nv_bfloat16* __restrict__ cache_v, const std::int8_t* __restrict__ cache_k_i8,
    const std::int8_t* __restrict__ cache_v_i8, const __half* __restrict__ cache_k_scale,
    const __half* __restrict__ cache_v_scale, const std::int32_t* __restrict__ positions,
    float scale, __nv_bfloat16* __restrict__ out, std::int32_t tokens,
    std::int32_t padded_context) {
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
    __nv_bfloat16* q_s = gqa_smem;     // [Br, D] swizzled
    __nv_bfloat16* k_s = q_s + Br * D; // [Bc, D] swizzled
    __nv_bfloat16* v_s = k_s + Bc * D; // [Bc, D] swizzled

    const int q_block  = static_cast<int>(blockIdx.x);
    const int q_head   = static_cast<int>(blockIdx.y);
    const int tid      = static_cast<int>(threadIdx.x);
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int q0       = q_block * Br;
    const int kv_head  = q_head / kGqaPrefillGroupSize;
    const int base_pos = positions[0];

    if (q_head >= kGqaPrefillQHeads || q0 >= tokens) { return; }

    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int a_mat     = lane >> 3;
    const int a_rin     = lane & 7;
    const int a_rowoff  = a_rin + ((a_mat & 1) << 3);
    const int b_rin     = lane & 7;
    const int b_koff    = ((lane >> 3) & 1) << 3;
    const int warp_row0 = warp * 16; // this warp owns rows [warp_row0, warp_row0+16)

    // Per-lane precomputed swizzled ldmatrix base addresses (see gqa_prefill_swz_addr).
    const unsigned q_sbase = gqa_prefill_smem_addr(q_s);
    const unsigned k_sbase = gqa_prefill_smem_addr(k_s);
    const unsigned v_sbase = gqa_prefill_smem_addr(v_s);
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
    // below); it stays resident for the whole key loop. Global Q rows are 256 bf16
    // contiguous, with a token stride of 256*QHeads.
    {
        constexpr int VecPerRow      = D / 8;
        constexpr int QRowStride     = D * kGqaPrefillQHeads; // global stride between tokens
        const __nv_bfloat16* q_block = q + gqa_prefill_q_index(q_head, 0, q0);
        if (q0 + Br <= tokens) {
#pragma unroll
            for (int chunk = tid; chunk < Br * VecPerRow; chunk += Threads) {
                const int row    = chunk >> 5;
                const int d      = (chunk & 31) << 3;
                __nv_bfloat16* p = &q_s[row * D + gqa_prefill_swz(row, d)];
                gqa_prefill_cp_async_cg_16(p, &q_block[row * QRowStride + d]);
            }
        } else {
#pragma unroll
            for (int chunk = tid; chunk < Br * VecPerRow; chunk += Threads) {
                const int row    = chunk >> 5;
                const int d      = (chunk & 31) << 3;
                __nv_bfloat16* p = &q_s[row * D + gqa_prefill_swz(row, d)];
                if (q0 + row < tokens) {
                    gqa_prefill_cp_async_cg_16(p, &q_block[row * QRowStride + d]);
                } else {
                    *reinterpret_cast<int4*>(p) = make_int4(0, 0, 0, 0);
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

    // Prologue: commit Q, then kick off K(0). The loop's wait<0> below drains both.
    qus::kernels::async_copy_commit();
    if constexpr (Quantized) {
        gqa_prefill_stage_kv_i8(k_s, cache_k_i8, cache_k_scale, kv_head, 0, max_query_abs,
                                padded_context, tid);
    } else {
        gqa_prefill_stage_kv(k_s, cache_k, kv_head, 0, max_query_abs, padded_context, tid);
    }
    qus::kernels::async_copy_commit();

    for (int kb = 0; kb < n_block_max; ++kb) {
        const int k0 = kb * Bc;

        qus::kernels::async_copy_wait<0>(); // K(kb) landed (also publishes q_s / prev PV done)
        __syncthreads();

        // Overlap V(kb) load against the QK MMA below.
        if constexpr (Quantized) {
            gqa_prefill_stage_kv_i8(v_s, cache_v_i8, cache_v_scale, kv_head, k0, max_query_abs,
                                    padded_context, tid);
        } else {
            gqa_prefill_stage_kv(v_s, cache_v, kv_head, k0, max_query_abs, padded_context, tid);
        }
        qus::kernels::async_copy_commit();

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
            gqa_prefill_ldmatrix_x4(af[0][0], af[0][1], af[0][2], af[0][3],
                                    gqa_prefill_swz_addr(q_lane_base, 0u, q_as, q_r));
#pragma unroll
            for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                gqa_prefill_ldmatrix_x4(
                    bf[0][nt2][0], bf[0][nt2][1], bf[0][nt2 + 1][0], bf[0][nt2 + 1][1],
                    gqa_prefill_swz_addr(k_lane_base + static_cast<unsigned>(nt2 * 4096), 0u, k_as,
                                         k_r));
            }
        }
#pragma unroll
        for (int k = 0; k < QKKs; ++k) {
            const int cur = k & 1;
            const int nxt = cur ^ 1;
            if (k + 1 < QKKs) {
                const unsigned ck = static_cast<unsigned>((k + 1) << 5);
                gqa_prefill_ldmatrix_x4(af[nxt][0], af[nxt][1], af[nxt][2], af[nxt][3],
                                        gqa_prefill_swz_addr(q_lane_base, ck, q_as, q_r));
#pragma unroll
                for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                    gqa_prefill_ldmatrix_x4(
                        bf[nxt][nt2][0], bf[nxt][nt2][1], bf[nxt][nt2 + 1][0], bf[nxt][nt2 + 1][1],
                        gqa_prefill_swz_addr(k_lane_base + static_cast<unsigned>(nt2 * 4096), ck,
                                             k_as, k_r));
                }
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                gqa_prefill_mma_m16n8k16_bf16(score[nt][0], score[nt][1], score[nt][2],
                                              score[nt][3], af[cur][0], af[cur][1], af[cur][2],
                                              af[cur][3], bf[cur][nt][0], bf[cur][nt][1]);
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
        bm0 = fmaxf(bm0, __shfl_xor_sync(FullMask, bm0, 1));
        bm0 = fmaxf(bm0, __shfl_xor_sync(FullMask, bm0, 2));
        bm1 = fmaxf(bm1, __shfl_xor_sync(FullMask, bm1, 1));
        bm1 = fmaxf(bm1, __shfl_xor_sync(FullMask, bm1, 2));

        const float nm0        = fmaxf(m0, bm0);
        const float nm1        = fmaxf(m1, bm1);
        const float nm0_scaled = nm0 * scale_l2;
        const float nm1_scaled = nm1 * scale_l2;
        const float alpha0     = gqa_prefill_exp2_fast(__fmaf_rn(m0, scale_l2, -nm0_scaled));
        const float alpha1     = gqa_prefill_exp2_fast(__fmaf_rn(m1, scale_l2, -nm1_scaled));

        // P = exp2(S - m), repacked into the PV A-fragment layout, plus local block row-sum.
        // The row-sum allreduce is deferred to the epilogue; only row max must be reduced per tile.
        float bl0 = 0.0f, bl1 = 0.0f;
        unsigned p_frag[PVKs][4];
        if (full_score_tile) {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const float p00 =
                    gqa_prefill_exp2_fast(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled));
                const float p01 =
                    gqa_prefill_exp2_fast(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled));
                const float p10 =
                    gqa_prefill_exp2_fast(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled));
                const float p11 =
                    gqa_prefill_exp2_fast(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled));
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
                const float p00 =
                    (score[nt][0] > -CUDART_INF_F)
                        ? gqa_prefill_exp2_fast(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled))
                        : 0.0f;
                const float p01 =
                    (score[nt][1] > -CUDART_INF_F)
                        ? gqa_prefill_exp2_fast(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled))
                        : 0.0f;
                const float p10 =
                    (score[nt][2] > -CUDART_INF_F)
                        ? gqa_prefill_exp2_fast(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled))
                        : 0.0f;
                const float p11 =
                    (score[nt][3] > -CUDART_INF_F)
                        ? gqa_prefill_exp2_fast(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled))
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

        qus::kernels::async_copy_wait<0>(); // V(kb) landed; QK done reading k_s
        __syncthreads();

        // Prefetch K(kb+1) into the (now-free) K buffer, overlapping the PV MMA.
        if (kb + 1 < n_block_max) {
            if constexpr (Quantized) {
                gqa_prefill_stage_kv_i8(k_s, cache_k_i8, cache_k_scale, kv_head, (kb + 1) * Bc,
                                        max_query_abs, padded_context, tid);
            } else {
                gqa_prefill_stage_kv(k_s, cache_k, kv_head, (kb + 1) * Bc, max_query_abs,
                                     padded_context, tid);
            }
            qus::kernels::async_copy_commit();
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
            gqa_prefill_ldmatrix_x4_trans(vf[0][0], vf[0][1], vf[0][2], vf[0][3],
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
                gqa_prefill_ldmatrix_x4_trans(
                    vf[nxt][0], vf[nxt][1], vf[nxt][2], vf[nxt][3],
                    gqa_prefill_swz_addr(v_lane_base + static_cast<unsigned>(k2 * 8192), ckv, v_as,
                                         v_r));
            }
            gqa_prefill_mma_m16n8k16_bf16(acc[n2][0], acc[n2][1], acc[n2][2], acc[n2][3],
                                          p_frag[k][0], p_frag[k][1], p_frag[k][2], p_frag[k][3],
                                          vf[cur][0], vf[cur][1]);
            gqa_prefill_mma_m16n8k16_bf16(acc[n2 + 1][0], acc[n2 + 1][1], acc[n2 + 1][2],
                                          acc[n2 + 1][3], p_frag[k][0], p_frag[k][1], p_frag[k][2],
                                          p_frag[k][3], vf[cur][2], vf[cur][3]);
        }
    }

    l0 += __shfl_xor_sync(FullMask, l0, 1);
    l0 += __shfl_xor_sync(FullMask, l0, 2);
    l1 += __shfl_xor_sync(FullMask, l1, 1);
    l1 += __shfl_xor_sync(FullMask, l1, 2);

    // Normalize once per row via reciprocal-multiply instead of 128 IEEE divides.
    const float inv_l0 = (l0 > 0.0f) ? __frcp_rn(l0) : 0.0f;
    const float inv_l1 = (l1 > 0.0f) ? __frcp_rn(l1) : 0.0f;
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
        const int d0    = n * 8 + 2 * lid;
        const int qrow0 = q0 + warp_row0 + gid;
        const int qrow1 = q0 + warp_row0 + gid + 8;
        if (qrow0 < tokens) {
            *reinterpret_cast<unsigned*>(&out[gqa_prefill_q_index(q_head, d0, qrow0)]) =
                gqa_prefill_pack_bf16(acc[n][0] * inv_l0, acc[n][1] * inv_l0);
        }
        if (qrow1 < tokens) {
            *reinterpret_cast<unsigned*>(&out[gqa_prefill_q_index(q_head, d0, qrow1)]) =
                gqa_prefill_pack_bf16(acc[n][2] * inv_l1, acc[n][3] * inv_l1);
        }
    }
}

} // namespace qus::kernels
