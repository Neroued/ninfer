#pragma once

// ninfer::kernels - fused A/B projections and GDN gate preparation.

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::kernels {

[[nodiscard]] std::size_t gdn_gating_proj_workspace_bytes(std::int32_t tokens);

void gdn_gating_proj(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                     const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws, Tensor& g,
                     Tensor& beta, cudaStream_t stream);

} // namespace ninfer::kernels
