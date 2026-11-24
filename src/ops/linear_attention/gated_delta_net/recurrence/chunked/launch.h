#pragma once

#include "core/device.h"
#include "ops/linear_attention/gated_delta_net/recurrence/chunked/workspace.h"

namespace ninfer::ops::detail::gated_delta_net::chunked {
void launch_prepare(const RecurrenceOperands& args, QkChunk* qk, ControlChunk* control,
                    bool normalize_qk, cudaStream_t stream);
void launch_recurrence(const RecurrenceOperands& args, const QkChunk* qk,
                       const ControlChunk* control, int value_tile, cudaStream_t stream);
} // namespace ninfer::ops::detail::gated_delta_net::chunked
