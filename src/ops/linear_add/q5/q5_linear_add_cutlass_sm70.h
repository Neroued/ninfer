#pragma once

// Volta (sm_70) tensor-core path for the q5 linear_add GEMM (MLP down-projection + residual,
// k=17408; attention input gate/value, k=6144). Same two-phase design as
// q4_linear_swiglu_cutlass_sm70: dequantize q5 groupwise-int weights (4-bit low nibble + 1-bit
// high plane) to a scratch FP16 buffer, cast bf16 activations to FP16, run CUTLASS's stock
// Sm70 Gemm with alpha=1/beta=1 so the epilogue reads the existing residual as C and writes
// the sum back to the same buffer as D (standard CUTLASS in-place accumulate). See
// the V100 implementation.

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

std::size_t q5_linear_add_cutlass_workspace_bytes(std::int32_t rows, std::int32_t k,
                                                   std::int32_t cols);

void q5_linear_add_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::ops::detail
