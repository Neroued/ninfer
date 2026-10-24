#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"

namespace ninfer::ops {

/**
 * Y[n,t] = sum_k W[n,k] * X[k,t] + bias[n].
 *
 * X [K,T], bias [N] and Y [N,T] are contiguous BF16; T>0. W is a registered
 * contiguous BF16 [N,K] geometry with matching logical/padded dimensions. Matrix
 * bases are 16-byte aligned; bias is naturally aligned. Uses A16 computation;
 * bias is added before the final BF16 output, without rounding the projection.
 * Y overlaps no input or weight storage. Inputs are preserved.
 * No workspace, hidden allocation or persistent state; supports CUDA Graph replay.
 */
void linear_bias(const Tensor& x, const Weight& weight, const Tensor& bias, Tensor& out,
                 CudaExecutionView execution);

} // namespace ninfer::ops
