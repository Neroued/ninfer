#pragma once

#include "ops/common/cuda/mma.cuh"
#include "ops/loop_message/rtx5090/kernel.cuh"

namespace ninfer::ops::detail::loop_message {
template <int K>
__device__ __forceinline__ int swizzle(int row, int col) {
    return row * K + (col ^ ((row & (K % 64 == 0 ? 7 : 3)) * 8));
}

template <int K>
__device__ __forceinline__ void load_a(const Bf16* tile, int row, int k, int lane,
                                       unsigned (&a)[4]) {
    const int r = row + (lane & 15), c = k + (lane >> 4) * 8;
    common::ldmatrix_x4(a[0], a[1], a[2], a[3], common::smem_addr(tile + swizzle<K>(r, c)));
}

template <int K>
__device__ __forceinline__ void load_b(const Bf16* tile, int row, int k, int lane,
                                       unsigned (&b)[2]) {
    const int r = row + (lane & 7), c = k + ((lane >> 3) & 1) * 8;
    common::ldmatrix_x2(b[0], b[1], common::smem_addr(tile + swizzle<K>(r, c)));
}

__device__ __forceinline__ void mma(float (&c)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    common::mma_bf16(c[0], c[1], c[2], c[3], a[0], a[1], a[2], a[3], b[0], b[1]);
}

} // namespace ninfer::ops::detail::loop_message
