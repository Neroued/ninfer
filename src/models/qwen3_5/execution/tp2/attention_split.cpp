#include "models/qwen3_5/execution/tp2/attention_split.h"

#include "models/qwen3_5/execution/linear.h"
#include "models/qwen3_5/execution/tp2/linear_split.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/tp2/attn_input_proj.h"

#include <array>
#include <stdexcept>
#include <variant>

namespace ninfer::models::qwen3_5::execution {

namespace {

const LinearParameters& split_projection(const AttentionParameters& parameters) {
    const auto* single = std::get_if<LinearParameters>(&parameters.projection);
    if (single == nullptr) {
        throw std::invalid_argument("tensor-parallel attention projection: a paired Q|K + gate|V "
                                    "projection has no two-device route");
    }
    return *single;
}

} // namespace

std::size_t attention_projection_split_workspace_bytes(const AttentionParameters& parameters,
                                                       std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("attention projection: invalid column interval");
    }
    const auto& single = split_projection(parameters);
    return ops::attn_input_proj_column_parallel_workspace_capacity_bytes(
        single.weight.qtype, single.policy, first, last);
}

void attention_projection_split(const std::array<Tensor, 2>& hidden,
                                const std::array<const AttentionParameters*, 2>& parameters,
                                const std::array<Tensor, 2>& query,
                                const std::array<Tensor, 2>& gate, const std::array<Tensor, 2>& key,
                                const std::array<Tensor, 2>& value,
                                const std::array<WorkspaceArena*, 2>& workspace,
                                const ExecutionContext& execution) {
    const std::array<const LinearParameters*, 2> shards{&split_projection(*parameters[0]),
                                                        &split_projection(*parameters[1])};
    const auto policy = split_policy(shards, "tensor-parallel attention projection");
    auto scope0       = workspace[0]->scope();
    auto scope1       = workspace[1]->scope();
    ops::attn_input_proj_column_parallel(hidden, split_weights(shards), query, gate, key, value,
                                         policy, workspace, execution);
}

} // namespace ninfer::models::qwen3_5::execution
