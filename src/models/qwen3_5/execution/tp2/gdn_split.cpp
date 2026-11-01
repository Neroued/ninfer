#include "models/qwen3_5/execution/tp2/gdn_split.h"

#include "models/qwen3_5/execution/linear.h"
#include "models/qwen3_5/execution/tp2/linear_split.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/tp2/gdn_gating_proj.h"
#include "ninfer/ops/tp2/gdn_input_proj.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <variant>

namespace ninfer::models::qwen3_5::execution {

namespace {

const LinearParameters& split_projection(const GdnParameters& parameters) {
    const auto* single = std::get_if<LinearParameters>(&parameters.projection);
    if (single == nullptr) {
        throw std::invalid_argument("tensor-parallel GDN projection: a paired Q|K + V|Z "
                                    "projection has no two-device route");
    }
    return *single;
}

std::array<const LinearParameters*, 2>
split_projections(const std::array<const GdnParameters*, 2>& parameters) {
    return {&split_projection(*parameters[0]), &split_projection(*parameters[1])};
}

} // namespace

std::size_t gdn_control_split_workspace_bytes(const GdnConfig& shard, std::int32_t hidden,
                                              std::int32_t first, std::int32_t last) {
    return ops::gdn_gating_proj_column_parallel_workspace_capacity_bytes(
        static_cast<std::int32_t>(shard.linear_num_value_heads), hidden, first, last);
}

std::size_t gdn_projection_split_workspace_bytes(const GdnParameters& parameters,
                                                 std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("GDN projection: invalid column interval");
    }
    const auto& w = split_projection(parameters);
    return ops::gdn_input_proj_column_parallel_workspace_capacity_bytes(
        w.weight.qtype, w.weight.n, w.weight.k, w.policy, first, last);
}

std::size_t gdn_snapshot_split_workspace_bytes(const GdnParameters& parameters, std::int32_t batch,
                                               std::int32_t first_width, std::int32_t last_width) {
    const auto& w = split_projection(parameters);
    // The native overlap contract takes a disjoint span even for a zero-scratch route.
    return std::max(
        std::size_t{1},
        ops::gdn_input_proj_conv_snapshot_column_parallel_workspace_capacity_bytes(
            w.weight.qtype, w.weight.n, w.weight.k, w.policy, batch, first_width, last_width));
}

std::size_t gdn_record_split_workspace_bytes(const GdnParameters& parameters, std::int32_t batch,
                                             std::int32_t first_width, std::int32_t last_width) {
    const auto& w = split_projection(parameters);
    // The native overlap contract takes a disjoint span even for a zero-scratch route.
    return std::max(
        std::size_t{1},
        ops::gdn_input_proj_conv_record_column_parallel_workspace_capacity_bytes(
            w.weight.qtype, w.weight.n, w.weight.k, w.policy, batch, first_width, last_width));
}

void gdn_control_split(const std::array<Tensor, 2>& hidden,
                       const std::array<const GdnParameters*, 2>& parameters,
                       const std::array<Tensor, 2>& g, const std::array<Tensor, 2>& beta,
                       const std::array<WorkspaceArena*, 2>& workspace,
                       const ExecutionContext& execution) {
    const GdnParameters& p0 = *parameters[0];
    const GdnParameters& p1 = *parameters[1];
    const std::array<Tensor, 2> a_log{p0.a_log, p1.a_log};
    const std::array<Tensor, 2> dt_bias{p0.dt_bias, p1.dt_bias};
    auto scope0         = workspace[0]->scope();
    auto scope1         = workspace[1]->scope();
    const auto* single0 = std::get_if<LinearParameters>(&p0.control);
    const auto* single1 = std::get_if<LinearParameters>(&p1.control);
    if (single0 != nullptr && single1 != nullptr) {
        ops::gdn_gating_proj_column_parallel(hidden, {single0->weight, single1->weight}, a_log,
                                             dt_bias, workspace, g, beta, execution);
        return;
    }
    const auto* pair0 = std::get_if<ops::PairedProjectionWeights>(&p0.control);
    const auto* pair1 = std::get_if<ops::PairedProjectionWeights>(&p1.control);
    if (pair0 == nullptr || pair1 == nullptr) {
        throw std::invalid_argument("tensor-parallel GDN control: ranks disagree on the A/B form");
    }
    ops::gdn_gating_proj_column_parallel(hidden, {pair0->first, pair1->first},
                                         {pair0->second, pair1->second}, a_log, dt_bias, workspace,
                                         g, beta, execution);
}

void gdn_projection_split(const std::array<Tensor, 2>& hidden,
                          const std::array<const GdnParameters*, 2>& parameters,
                          const std::array<Tensor, 2>& qkv, const std::array<Tensor, 2>& z,
                          const std::array<WorkspaceArena*, 2>& workspace,
                          const ExecutionContext& execution) {
    const auto shards = split_projections(parameters);
    const auto policy = split_policy(shards, "tensor-parallel GDN projection");
    auto scope0       = workspace[0]->scope();
    auto scope1       = workspace[1]->scope();
    ops::gdn_input_proj_column_parallel(hidden, split_weights(shards), qkv, z, policy, workspace,
                                        execution);
}

void gdn_projection_snapshot_split(
    const std::array<Tensor, 2>& hidden, const std::array<const GdnParameters*, 2>& parameters,
    const std::array<Tensor, 2>& conv_states, const std::array<Tensor, 2>& valid_columns,
    const std::array<Tensor, 2>& initial_slots, const std::array<Tensor, 2>& destination_slots,
    const std::array<Tensor, 2>& query, const std::array<Tensor, 2>& key,
    const std::array<Tensor, 2>& value, const std::array<Tensor, 2>& z,
    const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& execution) {
    const auto shards = split_projections(parameters);
    const auto policy = split_policy(shards, "tensor-parallel GDN snapshot projection");
    auto scope0       = workspace[0]->scope();
    auto scope1       = workspace[1]->scope();
    // As in the single-device form, each rank's Op receives exactly its disjoint scratch span.
    WorkspaceArena scratch0(workspace[0]->alloc_bytes(gdn_snapshot_split_workspace_bytes(
        *parameters[0], hidden[0].ne[2], hidden[0].ne[1], hidden[0].ne[1])));
    WorkspaceArena scratch1(workspace[1]->alloc_bytes(gdn_snapshot_split_workspace_bytes(
        *parameters[1], hidden[1].ne[2], hidden[1].ne[1], hidden[1].ne[1])));
    const std::array<Tensor, 2> convolution{parameters[0]->convolution, parameters[1]->convolution};
    ops::gdn_input_proj_conv_snapshot_column_parallel(
        hidden, split_weights(shards), convolution, conv_states, valid_columns, initial_slots,
        destination_slots, query, key, value, z, policy, {&scratch0, &scratch1}, execution);
}

void gdn_projection_record_split(
    const std::array<Tensor, 2>& hidden, const std::array<const GdnParameters*, 2>& parameters,
    const std::array<Tensor, 2>& conv_states, const std::array<Tensor, 2>& valid_columns,
    const std::array<Tensor, 2>& initial_slots, const std::array<Tensor, 2>& conv_record,
    const std::array<Tensor, 2>& query, const std::array<Tensor, 2>& key,
    const std::array<Tensor, 2>& value, const std::array<Tensor, 2>& z,
    const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& execution) {
    const auto shards = split_projections(parameters);
    const auto policy = split_policy(shards, "tensor-parallel GDN record projection");
    auto scope0       = workspace[0]->scope();
    auto scope1       = workspace[1]->scope();
    WorkspaceArena scratch0(workspace[0]->alloc_bytes(gdn_record_split_workspace_bytes(
        *parameters[0], hidden[0].ne[2], hidden[0].ne[1], hidden[0].ne[1])));
    WorkspaceArena scratch1(workspace[1]->alloc_bytes(gdn_record_split_workspace_bytes(
        *parameters[1], hidden[1].ne[2], hidden[1].ne[1], hidden[1].ne[1])));
    const std::array<Tensor, 2> convolution{parameters[0]->convolution, parameters[1]->convolution};
    ops::gdn_input_proj_conv_record_column_parallel(
        hidden, split_weights(shards), convolution, conv_states, valid_columns, initial_slots,
        conv_record, query, key, value, z, policy, {&scratch0, &scratch1}, execution);
}

} // namespace ninfer::models::qwen3_5::execution
