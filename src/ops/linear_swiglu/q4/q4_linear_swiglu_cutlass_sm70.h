#pragma once

// Volta (sm_70) tensor-core path for the q4 MLP gate/up GEMM, via NVIDIA CUTLASS's Sm70
// (FP16 mma.sync.m8n8k4) template GEMM. Two-phase: dequantize the q4 groupwise-int weights
// to a scratch FP16 buffer, cast bf16 activations to FP16, then run CUTLASS's stock Gemm
// with a direct bf16 epilogue output. Compute-bound prefill shapes only -- decode/small-T
// stays on the existing SIMT GemvPair/SmallTExact paths, which are already bandwidth-
// efficient and would pay the (T-independent) dequant cost for little benefit at small T.
// See docs/v100.md.

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

std::size_t q4_linear_swiglu_cutlass_workspace_bytes(std::int32_t gate_up_rows, std::int32_t k,
                                                      std::int32_t cols);

void q4_linear_swiglu_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& gate_up_out,
                                          WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::ops::detail
