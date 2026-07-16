#pragma once

#include "ops/common/math.cuh"
#include "ops/linear/codec/linear_codec.cuh"
#include "ops/linear/common/rowsplit_mma_common.cuh"
#include "core/tensor.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

void linear_rowsplit_q4_gate_up_silu_gemm_mma_launch(const Tensor& x, const Weight& weight,
                                                     Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
