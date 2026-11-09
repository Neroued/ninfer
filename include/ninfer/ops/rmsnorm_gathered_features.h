#pragma once

#include "core/device.h"
#include "core/tensor.h"

namespace ninfer::ops {

/**
 * RMSNorm over complete features in a rank-major all-gather result, without communication:
 *   x[r*2048+d,m] = gathered[d,m,r]
 *   out[h,m] = BF16(x[h,m]*gamma[h]/sqrt(sum_j x[j,m]^2/4096 + epsilon)).
 * gathered is contiguous BF16 [2048,M,2], gamma is contiguous BF16 [4096], and out is
 * contiguous BF16 [4096,M], M>0. Epsilon is positive and finite; gain is direct.
 * All operands are 16-byte aligned on execution.device(), with out disjoint from both
 * preserved inputs. There is no intermediate concat or normalized storage boundary.
 * The output is fully overwritten. No workspace, persistent state, hidden allocation
 * or synchronization. Graph replay consumes current inputs at the bound addresses.
 */
void rmsnorm_gathered_features(const Tensor& gathered, const Tensor& gamma, float epsilon,
                               Tensor& out, CudaExecutionView execution);

} // namespace ninfer::ops
