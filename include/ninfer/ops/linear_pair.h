#pragma once

#include "core/device.h"
#include "ninfer/ops/projection_weights.h"
#include "core/tensor.h"

namespace ninfer::ops {

// Capacity for every extent in the inclusive interval; invalid profiles or intervals throw.
[[nodiscard]] std::size_t linear_pair_workspace_capacity_bytes(const Weight& first_weight,
                                                               const Weight& second_weight,
                                                               std::int32_t min_tokens,
                                                               std::int32_t max_tokens);

[[nodiscard]] std::size_t linear_pair_workspace_capacity_bytes(LinearProfile first,
                                                               LinearProfile second,
                                                               std::int32_t min_tokens,
                                                               std::int32_t max_tokens);

/**
 * Y0 = decode(W0) * X0; Y1 = decode(W1) * X1, using the linear A16 arithmetic contract.
 * Inputs [K,T] and outputs [N,T] are contiguous, 16-byte-aligned BF16 matrices.
 * Weights have identical registered geometry/encoding: BF16 Contiguous or Q8_G32_FP16 RowSplit.
 * T is positive. Read-only inputs may alias. Outputs are disjoint from all operands.
 * Overwrites both outputs, uses no scratch and has no persistent state.
 */
void linear_pair(const Tensor& first_input, const Tensor& second_input, const Weight& first_weight,
                 const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                 CudaExecutionView execution);

} // namespace ninfer::ops
