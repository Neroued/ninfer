#pragma once

#include "core/device.h"
#include "models/qwen3_5/execution/parameters.h"
#include "ninfer/ops/softmax_attention.h"

#include <functional>

namespace ninfer::models::qwen3_5::execution {

struct CausalAttentionReadout;

using CausalAttentionSubmit = std::function<void(const CausalAttentionReadout&)>;

// Capture borrows the layer's operands and transient scratch on the execution stream.
struct CausalAttentionReadout {
    Tensor query, key, value, positions, valid_columns, table_rows, output;
    PagedKVBatchLayerView cache;
    ops::AttentionHeadGeometry geometry;
    float scale;
    MemorySpan scratch;

    [[nodiscard]] ops::OpGraphProfile profile(ops::ContextBucket, int sms) const;
    void execute(ops::ContextBucket, CudaExecutionView) const;
    void execute(ops::ContextBucket, MemoryArena&, CudaExecutionView) const;
    void submit(MemoryArena&, CudaExecutionView, const CausalAttentionSubmit&);
};

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
