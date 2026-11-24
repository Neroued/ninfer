#pragma once

#include "core/device.h"
#include "models/qwen3_5/execution/parameters.h"

namespace ninfer::models::qwen3_5::execution {

[[nodiscard]] std::size_t attention_projection_workspace_bytes(const AttentionProfile& parameters,
                                                               std::int32_t first,
                                                               std::int32_t last);
void attention_projection(const Tensor& hidden, const AttentionParameters& parameters,
                          Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                          MemoryArena& workspace, CudaExecutionView execution);

void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query,
               CudaExecutionView execution);
void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query, Tensor& key,
               CudaExecutionView execution);

} // namespace ninfer::models::qwen3_5::execution
