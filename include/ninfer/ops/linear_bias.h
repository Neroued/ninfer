#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"

namespace ninfer::ops {

/**
 * Y[n,t] = sum_k W[n,k] * X[k,t] + bias[n].
 *
 * X [K,T], bias [N] and Y [N,T] are contiguous BF16; T>0. W stores [N,K]
 * in its represented format. Matrix bases are 16-byte aligned; bias is naturally
 * aligned. Bias is added before the final BF16 output, without rounding the projection.
 * Y overlaps no input or weight storage. Inputs are preserved.
 * Policy controls permitted activation quantization. Workspace is caller-owned, call-scoped
 * and disjoint from all operands. No hidden allocation or persistent state; supports Graph replay.
 */
void linear_bias(const Tensor& x, const Weight& weight, const Tensor& bias, Tensor& out,
                 LinearPolicy policy, MemoryArena& workspace, CudaExecutionView execution);

// A16 execution requiring no workspace.
void linear_bias(const Tensor& x, const Weight& weight, const Tensor& bias, Tensor& out,
                 CudaExecutionView execution);

[[nodiscard]] std::size_t linear_bias_workspace_capacity_bytes(QType qtype, std::int32_t rows,
                                                               std::int32_t input_rows,
                                                               LinearPolicy policy,
                                                               std::int32_t min_tokens,
                                                               std::int32_t max_tokens);

} // namespace ninfer::ops
