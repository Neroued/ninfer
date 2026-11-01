#pragma once

#include "models/qwen3_5/execution/parameters.h"

namespace ninfer::models::qwen3_5::execution {

// The dense SwiGLU FFN added into `residual`: fused gate|up SwiGLU then linear_add, or with `mtp`
// (the Qwen3.5 MTP head) gate|up, silu_mul, down and a residual add.
[[nodiscard]] std::size_t dense_ffn_workspace_bytes(const DenseParameters& parameters,
                                                    std::int32_t first, std::int32_t last,
                                                    bool mtp = false);
void dense_ffn(const Tensor& hidden, const DenseParameters& parameters, Tensor& residual,
               WorkspaceArena& workspace, cudaStream_t stream, bool mtp = false);

} // namespace ninfer::models::qwen3_5::execution
