#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace ninfer::ops::detail {
struct AxialRope64Arguments {
    const std::int32_t* positions;
    const __nv_bfloat16 *q_weight, *k_weight;
    __nv_bfloat16 *q, *k;
    std::int64_t q_stride, k_stride;
    int tokens, q_heads, k_heads;
    float frequency_exponent, epsilon;
};

void launch_normalized_axial64(AxialRope64Arguments arguments, cudaStream_t stream);
} // namespace ninfer::ops::detail
