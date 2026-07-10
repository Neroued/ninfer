#pragma once

#include "kernels/linear/gemm/linear_rowsplit_gemm_mma.cuh"
#include "qus/core/tensor.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels::detail {

__device__ __forceinline__ float gate_up_silu_f32(float x) { return x / (1.0f + expf(-x)); }

void linear_rowsplit_q4_gate_up_silu_gemm_mma_launch(const Tensor& x, const Weight& weight,
                                                     Tensor& out, cudaStream_t stream);

} // namespace qus::kernels::detail
