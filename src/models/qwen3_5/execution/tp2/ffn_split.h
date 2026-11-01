#pragma once

// Two-device (tp 2) forms of the dense FFN and the MTP post-mixer FFN (ffn.h).
// Our file: the upstream header carries no hook.

#include "models/qwen3_5/execution/ffn.h"
#include "core/device.h"
#include "models/qwen3_5/execution/parameters.h"
#include "ninfer/ops/allreduce.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::models::qwen3_5::execution {

// Tensor-parallel Dense FFN over two ranks: column-parallel gate/up with SwiGLU into rank r's
// block of the intermediate activation, then the row-parallel down projection added to the
// replicated residual, which is the layer's second all-reduce. `parameters[r]` is rank r's shard
// over half the intermediate width. `exchange_lane` >= 0 only begins the down projection's
// all-reduce (project_add_row_parallel).
[[nodiscard]] std::size_t dense_ffn_split_workspace_bytes(const DenseParameters& parameters,
                                                          std::int32_t first, std::int32_t last);
void dense_ffn_split(const std::array<Tensor, 2>& hidden,
                     const std::array<const DenseParameters*, 2>& parameters,
                     const std::array<Tensor, 2>& residual, const std::array<Tensor, 2>& staging,
                     const std::array<WorkspaceArena*, 2>& workspace,
                     const ExecutionContext& execution, const ops::PeerEvents& events,
                     int exchange_lane = -1);

// Tensor-parallel form of the MTP post-mixer FFN, `dense_ffn(..., mtp = true)`: the column-parallel
// gate|up projection gives rank r its [gate_r; up_r] block, which it SiLU-multiplies into its block
// of the intermediate activation; the row-parallel down projection's all-reduce then leaves the
// complete delta on both ranks, added to the replicated residual. `staging[r]` matches
// `residual[r]`.
[[nodiscard]] std::size_t mtp_ffn_split_workspace_bytes(const DenseParameters& parameters,
                                                        std::int32_t first, std::int32_t last);
void mtp_ffn_split(const std::array<Tensor, 2>& hidden,
                   const std::array<const DenseParameters*, 2>& parameters,
                   const std::array<Tensor, 2>& residual, const std::array<Tensor, 2>& staging,
                   const std::array<WorkspaceArena*, 2>& workspace,
                   const ExecutionContext& execution, const ops::PeerEvents& events);

} // namespace ninfer::models::qwen3_5::execution
