#pragma once

// Volta (sm_70) tensor-core path for the fused attention input projection, via NVIDIA
// CUTLASS's Sm70 (FP16 mma.sync.m8n8k4) template GEMM. Same two-phase dequant-then-GEMM
// approach as q4_linear_swiglu_cutlass_sm70.h / q5_linear_add_cutlass_sm70.h, applied twice:
// query_key_weight (q4) and gate_value_weight (q5) each pack two logical outputs into one
// [7168, 5120] weight matrix (rows [0,6144) -> q/gate, rows [6144,7168) -> k/v). The dequant
// buffer is row-major-by-n (k-contiguous per row), so splitting by output is just a pointer
// offset into one shared dequant pass per weight -- no extra dequant work for the second
// output. Compute-bound prefill shapes only; decode/small-T stays on the existing SIMT
// gemv/split4/rowsplit paths. See docs/v100.md.

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

std::size_t q4_q5_attn_input_cutlass_workspace_bytes(std::int32_t cols);

void q4_q5_attn_input_cutlass_sm70_launch(const Tensor& x, const Weight& query_key_weight,
                                          const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                          Tensor& k, Tensor& v, WorkspaceArena& ws,
                                          cudaStream_t stream);

} // namespace ninfer::ops::detail
