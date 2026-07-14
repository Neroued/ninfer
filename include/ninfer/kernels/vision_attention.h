#pragma once

#include "ninfer/core/arena.h"
#include "ninfer/core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::kernels {

// Packed non-causal Vision MHA for Qwen3.6: 16 heads, D=72, BF16 inputs/output.
// cu_seqlens partitions P into independent full-attention segments.
void vision_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& cu_seqlens,
                      WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);

} // namespace ninfer::kernels
