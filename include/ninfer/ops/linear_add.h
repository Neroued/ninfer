#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include "core/device.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Capacity for every extent in the inclusive interval; invalid profiles or intervals throw.
[[nodiscard]] std::size_t linear_add_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                              std::int32_t input_rows,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

[[nodiscard]] std::size_t linear_add_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                              std::int32_t input_rows,
                                                              LinearPolicy policy,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

/**
 * residual[:,t] += W * x[:,t].
 *
 * x [K,T] and residual [N,T] are contiguous, 16-byte-aligned BF16 matrices, T > 0.
 * W [N,K] uses a registered BF16, Q4_G64_FP16, Q5_G64_FP16, Q8_G32_FP16,
 * FP8_E4M3FN_ROW_BF16 or NVFP4 geometry and its native stored representation.
 * x and weight storage must not overlap residual. Updates the entire residual in place.
 *
 * Q4/Q5/Q8/BF16 use A16. FP8 permits A8 with AllowA8/AllowA4; NVFP4 permits A4
 * with AllowA4. The overload without a policy uses A16Only.
 * Caller workspace is scoped to the call; A16 requires no scratch. No persistent state.
 */
void linear_add(const Tensor& x, const Weight& w, Tensor& residual, MemoryArena& ws,
                CudaExecutionView execution);

void linear_add(const Tensor& x, const Weight& w, Tensor& residual, LinearPolicy policy,
                MemoryArena& ws, CudaExecutionView execution);

} // namespace ninfer::ops
