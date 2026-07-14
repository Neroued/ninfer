#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "targets/qwen3_6_27b_rtx5090/impl/kernels/gqa_attention/gqa_attention.h"
#include "targets/qwen3_6_27b_rtx5090/impl/kernels/mtp/mtp.h"

#include <cuda_runtime.h>

#include <span>

namespace ninfer::targets::qwen3_6_27b_rtx5090::detail::schedule::detail {

// Applies only the visual columns covered by the shifted MTP input window. This is kept as a
// target-private schedule operation because the shift and chunk-boundary semantics belong to this
// exact checkpoint program, not to the generic scatter kernel.
void scatter_shifted_visual_embeddings(Tensor& input_embeddings, const Tensor& visual_embeddings,
                                       std::span<const std::int32_t> scatter_indices,
                                       std::int32_t shifted_begin, std::int32_t prompt_tokens,
                                       WorkspaceArena& work, cudaStream_t stream);

} // namespace ninfer::targets::qwen3_6_27b_rtx5090::detail::schedule::detail
