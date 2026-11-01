#pragma once

// Two-device (tp 2) forms of the Gated DeltaNet mixer projections (gdn.h).
// Our file: the upstream header carries no hook.

#include "models/qwen3_5/execution/gdn.h"
#include "core/device.h"
#include "models/qwen3_5/execution/parameters.h"
#include "ninfer/ops/allreduce.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::models::qwen3_5::execution {

// Tensor-parallel forms over two ranks. `parameters[r]` is rank r's shard: key heads
// [Hk/2*r, +Hk/2) and value heads [Hv/2*r, +Hv/2) of every Q|K|V|Z section in one contiguous
// parent, the matching convolution channels, and those value heads' A/B rows, A_log and dt_bias.
// The input RMSNorm is replicated elementwise work and runs per rank before gdn_control_split,
// because the fused norm+gating Op has no split form. The GDN core then runs per rank on the
// rank's heads with no communication. Only the single-parent input projection has a split route.
[[nodiscard]] std::size_t gdn_control_split_workspace_bytes(const GdnConfig& shard,
                                                            std::int32_t hidden, std::int32_t first,
                                                            std::int32_t last);
[[nodiscard]] std::size_t gdn_projection_split_workspace_bytes(const GdnParameters& parameters,
                                                               std::int32_t first,
                                                               std::int32_t last);
[[nodiscard]] std::size_t gdn_snapshot_split_workspace_bytes(const GdnParameters& parameters,
                                                             std::int32_t batch,
                                                             std::int32_t first_width,
                                                             std::int32_t last_width);
[[nodiscard]] std::size_t gdn_record_split_workspace_bytes(const GdnParameters& parameters,
                                                           std::int32_t batch,
                                                           std::int32_t first_width,
                                                           std::int32_t last_width);
void gdn_control_split(const std::array<Tensor, 2>& hidden,
                       const std::array<const GdnParameters*, 2>& parameters,
                       const std::array<Tensor, 2>& g, const std::array<Tensor, 2>& beta,
                       const std::array<WorkspaceArena*, 2>& workspace,
                       const ExecutionContext& execution);
void gdn_projection_split(const std::array<Tensor, 2>& hidden,
                          const std::array<const GdnParameters*, 2>& parameters,
                          const std::array<Tensor, 2>& qkv, const std::array<Tensor, 2>& z,
                          const std::array<WorkspaceArena*, 2>& workspace,
                          const ExecutionContext& execution);
void gdn_projection_snapshot_split(
    const std::array<Tensor, 2>& hidden, const std::array<const GdnParameters*, 2>& parameters,
    const std::array<Tensor, 2>& conv_states, const std::array<Tensor, 2>& valid_columns,
    const std::array<Tensor, 2>& initial_slots, const std::array<Tensor, 2>& destination_slots,
    const std::array<Tensor, 2>& query, const std::array<Tensor, 2>& key,
    const std::array<Tensor, 2>& value, const std::array<Tensor, 2>& z,
    const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& execution);
// Speculative verification form: each rank records its own heads' projected convolution inputs
// into `conv_record[r]` for the ReplaySSM fold instead of publishing a snapshot.
void gdn_projection_record_split(
    const std::array<Tensor, 2>& hidden, const std::array<const GdnParameters*, 2>& parameters,
    const std::array<Tensor, 2>& conv_states, const std::array<Tensor, 2>& valid_columns,
    const std::array<Tensor, 2>& initial_slots, const std::array<Tensor, 2>& conv_record,
    const std::array<Tensor, 2>& query, const std::array<Tensor, 2>& key,
    const std::array<Tensor, 2>& value, const std::array<Tensor, 2>& z,
    const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& execution);

} // namespace ninfer::models::qwen3_5::execution
