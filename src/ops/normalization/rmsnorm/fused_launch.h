#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ninfer::ops::detail::fused_rmsnorm {
struct Arguments {
    const __nv_bfloat16 *first, *second, *gain_first, *gain_second;
    const int* enabled;
    __nv_bfloat16 *out, *residual;
    int dimension, columns;
    float epsilon;
};

void launch_paired(Arguments arguments, bool vector_aligned, cudaStream_t stream);
void launch_residual(Arguments arguments, bool vector_aligned, cudaStream_t stream);
} // namespace ninfer::ops::detail::fused_rmsnorm
