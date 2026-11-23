#pragma once

#include "core/device.h"
#include "core/arena.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Computes one vocabulary argmax per column:
 *
 *   out[t] = min argmax_{0 <= v < valid_rows} float(logits[v,t]).
 *
 * `logits` is contiguous BF16 [physical_rows,T], `out` is contiguous I32 [T], and
 * 1 <= valid_rows <= physical_rows. Physical rows [valid_rows,physical_rows) do not
 * participate. Equal maxima select the lowest row index. `out` must not overlap `logits`.
 * The Op has no workspace and changes no state other than writing all of `out`.
 */
void argmax(const Tensor& logits, Tensor& out, std::int32_t valid_rows,
            CudaExecutionView execution);

// Scratch high-water mark for all positive extents through these bounds.
[[nodiscard]] std::size_t
argmax_with_probability_workspace_capacity_bytes(std::int32_t max_valid_rows,
                                                 std::int32_t max_columns);

// Same selection as argmax over finite consumed logits. Writes contiguous FP32 [T]
// with 1 / sum_v exp(float(logits[v,t]) - max_v float(logits[v,t])). All operands and
// caller-owned scratch are disjoint and on execution.device(). T == 0 is a shape-checked no-op.
void argmax_with_probability(const Tensor& logits, Tensor& out, Tensor& max_probabilities,
                             std::int32_t valid_rows, MemoryArena& workspace,
                             CudaExecutionView execution);

} // namespace ninfer::ops
