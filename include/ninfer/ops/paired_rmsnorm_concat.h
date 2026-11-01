#pragma once

#include "core/device.h"
#include "core/tensor.h"

namespace ninfer::ops {

/**
 * Independently normalize e and h, then concatenate their features:
 *   out[0:D,m] = enabled[m] ? e[:,m]*gamma_e/sqrt(mean(e[:,m]^2)+epsilon) : 0
 *   out[D:2D,m] = h[:,m]*gamma_h/sqrt(mean(h[:,m]^2)+epsilon).
 *
 * Inputs are contiguous BF16 [D,M], gains BF16 [D], and out BF16 [2D,M], with
 * D>0 and M>=0. enabled is contiguous Device I32 [M]; zero disables and nonzero enables. Disabled
 * embedding halves are positive zero without reading that embedding column.
 * Reductions and scaling use FP32; each normalized half is stored as BF16.
 * Gains are direct multipliers and epsilon is positive and finite.
 *
 * Inputs are preserved and may share read-only backing; out is disjoint from all
 * inputs. Operands are naturally aligned and belong to execution.device(). M=0
 * performs no device work. No workspace, persistent state or hidden allocation.
 * Stream submission and Graph replay consume current contents at fixed addresses.
 */
void paired_rmsnorm_concat(const Tensor& embedding, const Tensor& hidden,
                           const Tensor& embedding_weight, const Tensor& hidden_weight,
                           const Tensor& embedding_enabled, float epsilon, Tensor& out,
                           CudaExecutionView execution);

} // namespace ninfer::ops
