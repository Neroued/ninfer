#pragma once
#include "ops/mhc/kernel/residual.cuh"
#include <cuda_bf16.h>

namespace ninfer::ops::detail::mhc {
template <int Threads, int Values>
__global__ void post_vector_kernel(Operands o, std::int64_t tile_offset) {
    constexpr int HiddenTile = Threads * Values;
    const auto tile          = tile_offset + blockIdx.x;
    const auto m             = tile / (kHidden / HiddenTile);
    const int d              = (tile % (kHidden / HiddenTile)) * HiddenTile + threadIdx.x * Values;
    __shared__ float coeff[20];
    if (threadIdx.x < 4) coeff[threadIdx.x] = o.post[m * 4 + threadIdx.x];
    if (threadIdx.x < 16) coeff[4 + threadIdx.x] = o.mix[m * 16 + threadIdx.x];
    __syncthreads();
    common::Bf16Vector<Values> values[4];
    mixed_vectors(o, m, d, coeff, coeff + 4, values);
#pragma unroll
    for (int j = 0; j < 4; ++j)
        values[j].store(static_cast<__nv_bfloat16*>(o.updated) + m * kWidth + j * kHidden + d);
}

template <int Threads>
__global__ void post_kernel(Operands o, std::int64_t tile_offset) {
    const auto tile = tile_offset + blockIdx.x;
    const auto m    = tile / (kHidden / Threads);
    const int d     = (tile % (kHidden / Threads)) * Threads + threadIdx.x;
    __nv_bfloat16 values[4];
    residual_values<Form::Post>(o, m, d, values);
}
} // namespace ninfer::ops::detail::mhc
