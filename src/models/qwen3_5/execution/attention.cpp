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
