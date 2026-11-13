#pragma once
// Warp-specialized INT8-KV attention kernels share these device helpers: named barriers, the
// swizzled V tile layout and the exact INT8 -> FP16 conversion of V (the pipelined decode/verify
// kernel, grouped_pipe_mma.cuh, and the ping-pong prefill kernel, tiled_pp_mma.cuh).
#include <cuda_fp16.h>

namespace ninfer::ops::detail::int8_kv_pipe {

inline constexpr int kD = 256; // head dimension of every INT8-KV geometry

__device__ __forceinline__ void sync(int id, int threads) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

__device__ __forceinline__ void arrive(int id, int threads) {
    asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

// Byte offset of 16-byte chunk `chunk` of key row `key` in a V buffer: XOR with the key's low bits
// so the eight rows of one ldmatrix fall on distinct banks.
__device__ __forceinline__ int v_offset(int key, int chunk) {
    return key * kD + ((chunk ^ (key & 7)) << 4);
}

__device__ __forceinline__ void ldmatrix_x4_trans_b16(unsigned& r0, unsigned& r1, unsigned& r2,
                                                      unsigned& r3, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

// Two INT8 codes (bytes `sel` of `biased`, already XORed with 0x80) as an exact FP16 pair, times
// `scale`.
__device__ __forceinline__ unsigned v_pair(unsigned biased, unsigned sel, __half2 scale) {
    const unsigned bits = __byte_perm(biased, 0x64646464u, sel);
    const __half2 value =
        __hsub2(*reinterpret_cast<const __half2*>(&bits), __float2half2_rn(1152.0f));
    const __half2 scaled = __hmul2(value, scale);
    return *reinterpret_cast<const unsigned*>(&scaled);
}

} // namespace ninfer::ops::detail::int8_kv_pipe
