#pragma once

#include "core/device.h"
#include "core/tensor.h"

namespace ninfer::ops {

/**
 * For each column m, mixed[d,m] = sum_k FP32(alphas[k]) * FP32(inputs[d,k,m]).
 * out[d,m] = RN_BF16(mixed[d,m] * rsqrt(mean_d(mixed[d,m]^2) + eps) * FP32(weight[d])).
 * Each tap product rounds to FP32 before FP32 accumulation. The reduction consumes
 * the FP32 mix. Weight is a direct gain.
 *
 * Contiguous BF16 inputs [D,K,M], BF16 weight [D], BF16 out [D,M], and FP32 or BF16
 * alphas [K], with D>0, K>=1, M>=0 and finite eps>0. Nonempty operands are naturally
 * aligned and on the execution device. Out is disjoint from every input.
 * M=0 performs no device work and still validates alphas and weight. No workspace
 * or state. Graph replay reads current inputs.
 */
void weighted_sum_rmsnorm(const Tensor& inputs, const Tensor& alphas, const Tensor& weight,
                          float eps, Tensor& out, CudaExecutionView execution);

} // namespace ninfer::ops
