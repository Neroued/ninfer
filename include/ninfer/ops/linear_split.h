#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/projection_weights.h"

#include "core/device.h"
#include <span>

namespace ninfer::ops {

/**
 * Y_s[r,t] = sum_k decode(W)[begin_s+r,k] * X[k,t], with begin_s equal to the
 * sum of preceding output row counts. For paired weights, W is the logical row
 * concatenation [first; second]; an output segment may cross that parent boundary.
 * No physical weight concatenation or packed parent output is materialized.
 *
 * X is contiguous BF16 [K,T]; outputs are ordered, contiguous BF16 [N_s,T], with positive row
 * counts summing to the weight row count. T>0; activation/output bases are 16-byte
 * aligned. Outputs are mutually disjoint and overlap neither X nor any weight plane.
 * Inputs and weights are preserved. Logical and padded weight dimensions agree.
 *
 * Registered forms include BF16, Q8-G32/FP16, FP8 row-scale, NVFP4 block-scale and
 * paired Q4-G64/FP16 + Q5-G64/FP16. Single-parent activation permissions follow
 * LinearPolicy; paired forms use A16.
 * Workspace is caller-owned, call-scoped and disjoint from all operands. The Op
 * allocates no hidden device memory, owns no persistent state and supports Graph replay.
 */
void linear_split(const Tensor& x, const ProjectionWeights& weights,
                  std::span<const Tensor> outputs, MemoryArena& workspace,
                  CudaExecutionView execution);

/** Single-parent A16 computation with no transient workspace. */
void linear_split(const Tensor& x, const Weight& weight, std::span<const Tensor> outputs,
                  CudaExecutionView execution);

/**
 * Y_s[r,t] = sum_k W[begin_s+r,k] * X[k,t] + bias[begin_s+r].
 *
 * Single-parent BF16/A16 form with contiguous, naturally aligned BF16 bias [N].
 * Bias is added before output rounding. Output mapping, alignment and disjointness
 * follow linear_split; outputs also overlap no bias storage. Supports registered
 * BF16 geometries/segmentations, without workspace or hidden allocations.
 */
void linear_split(const Tensor& x, const Weight& weight, const Tensor& bias,
                  std::span<const Tensor> outputs, CudaExecutionView execution);

/**
 * Minimum call-local capacity covering inclusive [min_columns,max_columns], both positive.
 * Output row counts select the same registered segmentation as execution. Inspects only
 * host-visible format, shape and activation permission; no device backing is required.
 * Capacity starts at a 256-byte-aligned arena cursor and excludes trailing padding.
 */
[[nodiscard]] std::size_t
linear_split_workspace_capacity_bytes(const ProjectionProfile& weights,
                                      std::span<const std::int32_t> output_rows,
                                      std::int32_t min_columns, std::int32_t max_columns);

[[nodiscard]] inline std::size_t
linear_split_workspace_capacity_bytes(const ProjectionWeights& weights,
                                      std::span<const std::int32_t> rows, std::int32_t first,
                                      std::int32_t last) {
    return linear_split_workspace_capacity_bytes(projection_profile(weights), rows, first, last);
}
} // namespace ninfer::ops
