#pragma once

// Fixed Qwen3.6 GQA geometry and leaf PTX helpers shared by the independently tuned
// BF16 and INT8 prompt kernels. This file deliberately owns no staging policy,
// shared-memory arena, warp schedule, or kernel body.

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels {

inline constexpr int kGqaPrefillHeadDim   = 256;
inline constexpr int kGqaPrefillQHeads    = 24;
inline constexpr int kGqaPrefillKVHeads   = 4;
inline constexpr int kGqaPrefillGroupSize = 6;

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

// XOR-swizzled b16 element address. INT8 operands use the same layout by packing
// two consecutive signed bytes into each b16 lane before ldmatrix.
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

__device__ __forceinline__ void gqa_prefill_cp_async_cg_16(void* smem_dst,
                                                            const void* gmem_src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : :
                 "r"(static_cast<unsigned>(__cvta_generic_to_shared(smem_dst))), "l"(gmem_src));
}

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

__device__ __forceinline__ void gqa_prefill_ldmatrix_x2(unsigned& a0, unsigned& a1,
                                                        unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(a0), "=r"(a1)
                 : "r"(addr));
}

__device__ __forceinline__ void gqa_prefill_ldmatrix_x4_trans(unsigned& b0, unsigned& b1,
                                                              unsigned& b2, unsigned& b3,
                                                              unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3)
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

__device__ __forceinline__ void gqa_prefill_mma_m16n8k32_s8(int& c0, int& c1, int& c2, int& c3,
                                                            unsigned a0, unsigned a1, unsigned a2,
                                                            unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float gqa_prefill_exp2_fast(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

} // namespace qus::kernels
