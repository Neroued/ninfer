#pragma once

// Volta (sm_70) tensor-core path for the fused GDN (Gated DeltaNet) input projection, via
// NVIDIA CUTLASS's Sm70 (FP16 mma.sync.m8n8k4) template GEMM. Same two-phase dequant-then-GEMM
// approach as the other three CUTLASS conversions in this port: qk_weight (q4, [4096,5120]) is
// one plain GEMM into qkv[0:4096]; value_z_weight (q5, [12288,5120]) packs two logical outputs
// (rows [0,6144) -> value, into qkv[4096:10240); rows [6144,12288) -> z, a separate tensor) --
// the dequant buffer is row-major-by-n (k-contiguous per row), so splitting is a pointer offset
// into one shared dequant pass, not a second dequant. Compute-bound prefill shapes only;
// decode/small-T stays on the existing SIMT/gemv/split4 paths. See docs/v100.md.

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

std::size_t q4_q5_gdn_input_cutlass_workspace_bytes(std::int32_t cols);

void q4_q5_gdn_input_cutlass_sm70_launch(const Tensor& x, const Weight& qk_weight,
                                         const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                                         WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::ops::detail
