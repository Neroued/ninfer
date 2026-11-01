#pragma once

// Two-device (tp 2) forms of the full-attention projections (attention.h).
// Our file: the upstream header carries no hook.

#include "models/qwen3_5/execution/attention.h"
#include "core/device.h"
#include "models/qwen3_5/execution/parameters.h"
#include "ninfer/ops/allreduce.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::models::qwen3_5::execution {

// Tensor-parallel forms over two ranks. `parameters[r]` is rank r's shard: 1/2 of the query/gate
// heads and of the KV heads, stacked query|key|gate|value in one contiguous parent. Only that
// single-parent form has a split route; a paired Q|K + gate|V projection is rejected.
[[nodiscard]] std::size_t
attention_projection_split_workspace_bytes(const AttentionParameters& parameters,
                                           std::int32_t first, std::int32_t last);
void attention_projection_split(const std::array<Tensor, 2>& hidden,
                                const std::array<const AttentionParameters*, 2>& parameters,
                                const std::array<Tensor, 2>& query,
                                const std::array<Tensor, 2>& gate, const std::array<Tensor, 2>& key,
                                const std::array<Tensor, 2>& value,
                                const std::array<WorkspaceArena*, 2>& workspace,
                                const ExecutionContext& execution);

} // namespace ninfer::models::qwen3_5::execution
