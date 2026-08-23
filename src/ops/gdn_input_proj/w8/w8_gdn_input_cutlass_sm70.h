#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Volta (sm_70) tensor-core path for the 35B-A3B W8 GDN input projection.
//
// Measured at 41% of a3b prefill GPU time on its own (on V100): the
// SIMT row-view split that made a3b run leaves the largest GEMM in the model on
// a ~1 TFLOP/s kernel. Same two-phase treatment as the four q4/q5 conversions
// already in this port -- dequantize the parent into an FP16 scratch buffer,
// cast activations, then run CUTLASS's Sm70 tensor-core GEMM.
[[nodiscard]] std::size_t w8_gdn_input_cutlass_workspace_bytes(std::int32_t cols);

void w8_gdn_input_cutlass_sm70_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                      WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::ops::detail
