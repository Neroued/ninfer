#include "models/qwen3_5/execution/attention.h"

#include "ninfer/ops/linear_split.h"
#include "ninfer/ops/rope.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {
namespace {

void require_rope_axes(const Tensor& positions, const RopeConfig& config) {
    if (positions.ne[1] != 3) { return; }
    for (std::size_t i = 0; i < config.pair_axes.size(); ++i) {
        if (config.pair_axes[i] != i % 3) {
            throw std::invalid_argument("text RoPE: this MRoPE axis mapping has no native route");
        }
    }
}

} // namespace

ops::OpGraphProfile CausalAttentionReadout::profile(ops::ContextBucket bucket, int sms) const {
    return ops::causal_softmax_attention_graph_profile(geometry, cache.storage, bucket, query.ne[3],
                                                       query.ne[2], sms, true,
                                                       valid_columns.data != nullptr);
}

void CausalAttentionReadout::execute(ops::ContextBucket bucket, MemoryArena& workspace,
                                     CudaExecutionView cuda) const {
    auto out = output;
    ops::causal_softmax_attention(query, key, value, positions, valid_columns, table_rows, geometry,
                                  scale, cache, bucket, workspace, out, cuda);
}

void CausalAttentionReadout::execute(ops::ContextBucket bucket, CudaExecutionView cuda) const {
    MemoryArena workspace(scratch);
    execute(bucket, workspace, cuda);
}

void CausalAttentionReadout::submit(MemoryArena& workspace, CudaExecutionView cuda,
                                    const CausalAttentionSubmit& submit) {
    // All bucket boundaries are page-aligned, so rounding to whole pages preserves the range.
    const auto pages   = static_cast<std::uint32_t>(cache.k_pages.records.page_offsets.ne[0]);
    const auto buckets = ops::ContextBucketRange::through(pages * kPagedKVPageSize);
    const auto bytes   = ops::causal_softmax_attention_workspace_capacity_bytes(
        geometry, cache.storage, buckets, query.ne[3], query.ne[2], query.ne[2],
        cuda.multiprocessor_count());
    auto scope = workspace.scope();
    scratch    = bytes ? workspace.alloc_bytes(bytes) : MemorySpan{nullptr, 0, cuda.device()};
    submit(*this);
}

std::size_t attention_projection_workspace_bytes(const AttentionProfile& parameters,
                                                 std::int32_t first, std::int32_t last) {
    return ops::linear_split_workspace_capacity_bytes(parameters.projection,
                                                      parameters.projection_rows, first, last);
}

void attention_projection(const Tensor& hidden, const AttentionParameters& parameters,
                          Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                          MemoryArena& workspace, CudaExecutionView execution) {
    std::array outputs{query, key, gate, value};
    ops::linear_split(hidden, parameters.projection, outputs, workspace, execution);
}

void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query,
               CudaExecutionView execution) {
    require_rope_axes(positions, config);
    ops::rope(positions, dimension(config.rotary_dim), config.rope_theta, query, execution);
}

void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query, Tensor& key,
               CudaExecutionView execution) {
    require_rope_axes(positions, config);
    ops::rope(positions, dimension(config.rotary_dim), config.rope_theta, query, key, execution);
}

} // namespace ninfer::models::qwen3_5::execution
