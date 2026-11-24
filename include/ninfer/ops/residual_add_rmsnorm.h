#pragma once

#include "core/device.h"
#include "core/tensor.h"

namespace ninfer::ops {

/**
 * r = RN_BF16(FP32(residual) + FP32(branch));
 * normalized = RN_BF16(FP32(r)*weight/sqrt(mean_d(FP32(r)^2)+epsilon)).
 * Writes both r and normalized; the norm consumes the actual published BF16 r.
 * Weight is a direct gain, with FP32 reduction and scaling.
 *
 * Inputs/outputs are contiguous BF16 [D,M], D>0, M>=0; weight is BF16 [D].
 * Epsilon is positive and finite. All operands are naturally aligned and on the
 * execution device. residual_out may exactly alias residual; every other output
 * overlap is forbidden, including branch overlapping an in-place residual_out.
 * Read-only inputs may share backing and are preserved except for that exact alias.
 *
 * No workspace, persistent state or hidden allocation. M=0 performs no device work.
 * Stream submission and Graph replay consume current contents at fixed addresses.
 */
void residual_add_rmsnorm(const Tensor& residual, const Tensor& branch, const Tensor& weight,
                          float epsilon, Tensor& residual_out, Tensor& normalized_out,
                          CudaExecutionView execution);

} // namespace ninfer::ops
