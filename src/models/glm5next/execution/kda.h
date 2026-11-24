#pragma once
#include "core/linear_attention_state.h"
#include "models/glm5next/execution/batch.h"
#include "models/glm5next/execution/parameters.h"

namespace ninfer::models::glm5next::execution {
void kda(const Tensor& input, const ModelProfile&, const KdaParameters&, const BatchView&,
         const LinearAttentionStateView&, std::uint32_t layer, const StateUpdate&, Tensor& output,
         MemoryArena&, CudaExecutionView);
} // namespace ninfer::models::glm5next::execution
