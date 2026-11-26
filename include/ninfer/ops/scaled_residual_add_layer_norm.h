#pragma once

#include "core/device.h"
#include "core/tensor.h"

namespace ninfer::ops {

/**
 * residual_out = RN_BF16(FP32(residual) + FP32(RN_BF16(FP32(update) * FP32(gamma)))).
 * Residual, update, and residual_out are contiguous BF16 [D,M], D>0, M>=0.
 * Gamma is contiguous BF16 or FP32 [D] and broadcasts over M. Residual_out may
 * exactly alias residual. Every other output overlap is forbidden.
 *
 * Nonempty operands are naturally aligned and on the execution device. M=0 performs
 * no device work and still validates gamma. No workspace or state. Graph replay
 * reads current inputs.
 */
void scaled_residual_add(const Tensor& residual, const Tensor& update, const Tensor& gamma,
                         Tensor& residual_out, CudaExecutionView execution);

/**
 * Also writes normalized_out[d,m] = RN_BF16((r[d,m] - mean_d(r)) *
 * rsqrt(mean_d((r - mean_d(r))^2) + eps) * FP32(weight[d]) + FP32(bias[d])),
 * where r widens the published BF16 residual_out. Weight and bias are BF16 [D],
 * normalized_out is BF16 [D,M], and eps is positive and finite. These operands
 * follow the same layout and device requirements. Normalized_out is disjoint from
 * every input and residual_out. Residual_out also cannot overlap weight or bias.
 * M=0 also validates weight and bias.
 */
void scaled_residual_add_layer_norm(const Tensor& residual, const Tensor& update,
                                    const Tensor& gamma, const Tensor& weight, const Tensor& bias,
                                    float eps, Tensor& residual_out, Tensor& normalized_out,
                                    CudaExecutionView execution);

} // namespace ninfer::ops
