#pragma once

#include "models/qwen4_exp/execution/batch.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "core/linear_attention_state.h"

namespace ninfer::models::qwen4_exp::execution {

void gdn(const Tensor& input, const GdnConfig& c, const GdnParameters&, const BatchView&,
         const LinearAttentionStateView&, std::uint32_t layer, const StateUpdate&, float epsilon,
         Tensor& output, MemoryArena&, CudaExecutionView);

} // namespace ninfer::models::qwen4_exp::execution
