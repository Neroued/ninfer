#pragma once

// Volta (sm_70) tensor-core primitives for GQA attention: the mma.sync.m8n8k4 QK^T/PV
// instructions and their warp-level fragment addressing, transcribed faithfully from
// llama.cpp's ggml-cuda/mma.cuh (MIT licensed). The formulas are checked against an independent
// host oracle, and semantic operand roles follow llama.cpp's Volta fattn-mma-f16.cuh call sites
// rather than being inferred from tile shape alone.
//
// Fragment shapes (all per-warp, 8 registers or fewer per thread):
//   Q  (QK^T "A" operand): 32 rows x 8 real k-elems, 4 half2 regs/thread, I-major.
//   K  (QK^T "B" operand):  8 rows x 8 real k-elems, 4 half2 regs/thread, I-major-mirrored
//                           (each K row's data is replicated across 4 groups of 8 lanes).
//   D  (QK^T output):      32 rows x 8 cols, 8 float regs/thread, I-major.
//   P  (PV "A" operand, after softmax + float->half2 conversion): same layout as Q's tile.
//   V  (PV "B" operand):    8 rows x 8 real d-elems, 4 half2 regs/thread, J-major-mirrored.
//   PV (PV output):        32 rows x 8 cols, 4 half2 regs/thread, same layout as Q's tile.

#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ninfer::ops {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

// --- Fragment addressing -------------------------------------------------------------

// Q / P / PV-output: tile<32,4,half2 or float,I_MAJOR>. l in [0,4) for half2, [0,8) unused here.
__device__ __forceinline__ int volta_qp_get_i() { return threadIdx.x & 31; }
__device__ __forceinline__ int volta_qp_get_j(int l) { return l; }

// K: tile<8,4,half2,I_MAJOR_MIRRORED>.
__device__ __forceinline__ int volta_k_get_i() {
    const int lane = threadIdx.x & 31;
    return ((lane / 16) * 4) + (lane % 4);
}
__device__ __forceinline__ int volta_k_get_j(int l) { return l; }

// D (QK^T output float accumulator): tile<32,8,float,I_MAJOR>. l in [0,8).
__device__ __forceinline__ int volta_d_get_i(int l) {
    const int lane = threadIdx.x & 31;
    return (l & 2) + (lane & ~2);
}
__device__ __forceinline__ int volta_d_get_j(int l) {
    const int lane = threadIdx.x & 31;
    return (lane & 2) + (l & (4 + 1));
}

// V: tile<8,4,half2,J_MAJOR_MIRRORED>. l in [0,4).
__device__ __forceinline__ int volta_v_get_i(int l) {
    const int lane = threadIdx.x & 31;
    return ((l / 2) * 4) + (lane % 4);
}
__device__ __forceinline__ int volta_v_get_j(int l) {
    const int lane = threadIdx.x & 31;
    return ((lane / 16) * 2) + (l % 2);
}

// --- Loads (plain addressed loads -- Volta has no ldmatrix, sm_75+ only) -------------------

// Loads a full 4-half2 row for Q or P-shaped tiles (get_j(l)=l for all l, so one row covers
// all 4 registers). `stride` is in half2 units.
__device__ __forceinline__ void volta_load_qp(half2 (&dst)[4], const half2* __restrict__ base,
                                              int stride) {
    const int row = volta_qp_get_i();
#pragma unroll
    for (int l = 0; l < 4; ++l) { dst[l] = base[row * stride + l]; }
}

__device__ __forceinline__ void volta_load_k(half2 (&dst)[4], const half2* __restrict__ base,
                                             int stride) {
    const int row = volta_k_get_i();
#pragma unroll
    for (int l = 0; l < 4; ++l) { dst[l] = base[row * stride + l]; }
}

__device__ __forceinline__ void volta_load_v(half2 (&dst)[4], const half2* __restrict__ base,
                                             int stride) {
#pragma unroll
    for (int l = 0; l < 4; ++l) { dst[l] = base[volta_v_get_i(l) * stride + volta_v_get_j(l)]; }
}

// --- mma.sync.m8n8k4 wrappers ---------------------------------------------------------------

// QK^T: D[32x8 float] += Q[32x8 half, as A] @ K[8x8 half, as B]^T. K real k-dim per call = 8.
__device__ __forceinline__ void volta_mma_qk(float (&d)[8], const half2 (&q)[4],
                                             const half2 (&k)[4]) {
    const int* Axi = reinterpret_cast<const int*>(q);
    const int* Bxi = reinterpret_cast<const int*>(k);
    int* Dxi        = reinterpret_cast<int*>(d);
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
                 "{%0, %1, %2, %3, %4, %5, %6, %7};"
                 : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3]), "+r"(Dxi[4]),
                   "+r"(Dxi[5]), "+r"(Dxi[6]), "+r"(Dxi[7])
                 : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[0]), "r"(Bxi[1]));
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
                 "{%0, %1, %2, %3, %4, %5, %6, %7};"
                 : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3]), "+r"(Dxi[4]),
                   "+r"(Dxi[5]), "+r"(Dxi[6]), "+r"(Dxi[7])
                 : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[2]), "r"(Bxi[3]));
}

// Raw-operand form of the same instruction, for the quadpair-split-N mapping (see
// q4_volta_qpn_gemm.cuh). volta_mma_qk above passes one A and one B fragment that every quadpair
// shares; the QPN mapping gives each quadpair its own B and its own accumulator half, so its
// operands are per-quadpair registers rather than warp-wide fragments and cannot go through the
// fragment-array signature. Identical instruction, one slice (4 real k) per call.
__device__ __forceinline__ void volta_mma_qp_n(float (&d)[8], unsigned a0, unsigned a1,
                                               unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
                 "{%0, %1, %2, %3, %4, %5, %6, %7};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
                   "+f"(d[6]), "+f"(d[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// PV: PV[32x8 half, as output] += P[32x8 half, as A] @ V[8x8 half, as B]. Real k per call = 8.
__device__ __forceinline__ void volta_mma_pv(half2 (&pv)[4], const half2 (&p)[4],
                                             const half2 (&v)[4]) {
    const int* Pxi = reinterpret_cast<const int*>(p);
    const int* Vxi = reinterpret_cast<const int*>(v);
    int* PVxi        = reinterpret_cast<int*>(pv);
    asm volatile("mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16 "
                 "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3};"
                 : "+r"(PVxi[0]), "+r"(PVxi[1]), "+r"(PVxi[2]), "+r"(PVxi[3])
                 : "r"(Pxi[0]), "r"(Pxi[1]), "r"(Vxi[0]), "r"(Vxi[1]));
    asm volatile("mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16 "
                 "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3};"
                 : "+r"(PVxi[0]), "+r"(PVxi[1]), "+r"(PVxi[2]), "+r"(PVxi[3])
                 : "r"(Pxi[2]), "r"(Pxi[3]), "r"(Vxi[2]), "r"(Vxi[3]));
}

// Convert the QK^T float D-tile (post-softmax, i.e. P) directly into the half2 layout the PV
// mma's "A" operand needs -- register-only, one warp shuffle, no shared memory (Volta-
// specific; Turing+ needs an actual transpose here, Volta doesn't -- see docs/v100.md).
__device__ __forceinline__ void volta_softmax_to_half2(half2 (&p)[4], const float (&d)[8]) {
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 4) {
        p[l0 / 2 + 0] = __floats2half2_rn(d[l0 + 0], d[l0 + 1]);
        p[l0 / 2 + 1] = __floats2half2_rn(d[l0 + 2], d[l0 + 3]);
        const int lane      = threadIdx.x & 31;
        const int swap_idx = l0 / 2 + (((lane % 4) / 2) ^ 1);
        p[swap_idx]          = __shfl_xor_sync(0xFFFFFFFFu, p[swap_idx], 2, 32);
    }
}

#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

} // namespace ninfer::ops
