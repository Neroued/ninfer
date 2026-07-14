#pragma once

#include "kernels/common/math.cuh"
#include "kernels/linear/gemm/linear_rowsplit_gemm_mma.cuh"
#include "core/tensor.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::kernels::detail {

void linear_rowsplit_q4_gate_up_silu_gemm_mma_launch(const Tensor& x, const Weight& weight,
                                                     Tensor& out, cudaStream_t stream);

} // namespace ninfer::kernels::detail
