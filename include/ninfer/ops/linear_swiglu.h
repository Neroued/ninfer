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
[[nodiscard]] std::size_t linear_swiglu_workspace_capacity_bytes(QType qtype,
                                                                 std::int32_t gate_up_rows,
                                                                 std::int32_t input_rows,
                                                                 std::int32_t min_tokens,
                                                                 std::int32_t max_tokens);

[[nodiscard]] std::size_t
linear_swiglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                       std::int32_t input_rows, LinearPolicy policy,
                                       std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * out[i,t] = SiLU((W_gate * x)[i,t]) * (W_up * x)[i,t].
 *
 * x [K,T] and out [M,T] are contiguous, 16-byte-aligned BF16 matrices.
 * gate_up_weight [2*M,K] stores all M gate rows followed by their matching M up rows.
 * Registered BF16, Q4_G64_FP16, Q8_G32_FP16, FP8_E4M3FN_ROW_BF16 and NVFP4 geometries
 * use their native stored representation. Inputs and weight storage must not overlap out.
 * Writes the entire output and has no persistent state.
 *
 * BF16/Q4/Q8 use A16 under every policy. FP8 permits A8 with AllowA8/AllowA4.
 * NVFP4 permits A4 with AllowA4; A16Only/AllowA8 support only T <= 16 for NVFP4.
 * Otherwise every positive T is admitted. The overload without a policy uses A16Only.
 * Caller workspace covers the complete operation and is scoped to the call.
 */
void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, LinearPolicy policy,
                   MemoryArena& ws, CudaExecutionView execution);

void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, MemoryArena& ws,
                   CudaExecutionView execution);

// Explicit clamped product: SiLU(min(gate, limit)) * clamp(up, -limit, limit).
// limit must be finite and positive. Clamping is supported by registered BF16, Q8-G32/FP16 and
// NVFP4 geometries; unsupported combinations throw. Gate/up accumulators feed the product without
// an intermediate BF16 rounding boundary. Other operands/effects follow linear_swiglu.
void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, float limit,
                   LinearPolicy policy, MemoryArena& ws, CudaExecutionView execution);

/**
 * g = W_gate*x + bias_gate; u = W_up*x + bias_up;
 * out[i,t] = SiLU(min(g[i,t],limit)) * clamp(u[i,t],-limit,limit).
 *
 * Registered contiguous BF16 [2*M,K] weights and naturally aligned BF16 bias [2*M]
 * both use [gate;up] row order. Uses A16 computation with finite positive limit.
 * The projection and bias feed the product without intermediate BF16 rounding.
 * x/out follow linear_swiglu's matrix and alias requirements; out also overlaps no bias.
 * Inputs are preserved. No workspace, hidden allocation or persistent state;
 * supports CUDA Graph replay.
 */
void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, const Tensor& bias, Tensor& out,
                   float limit, CudaExecutionView execution);

} // namespace ninfer::ops
