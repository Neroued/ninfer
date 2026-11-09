#pragma once

#include "core/device.h"
#include "core/tensor.h"

namespace ninfer::ops {

/**
 * output[d,(W-1)*b+i-1] = input[d,i,b] * weight[d]
 *                        / sqrt(sum_j input[j,i,b]^2 / D + epsilon), for 1 <= i < W.
 *
 * Input is BF16 [D,W,B], weight is BF16 [D], and output is BF16 [D,(W-1)*B], with
 * D=4096 or 5120, W=2..16 and B=1..8. Epsilon is positive and finite.
 * All tensors are contiguous, non-null, 16-byte aligned and
 * pairwise disjoint. Uses plain multiplicative gain. Anchor columns input[:,0,b] are not read.
 * Writes all of output and preserves inputs. All operands belong to execution.device().
 * Graph replay consumes current input values. No workspace or state is used.
 */
void rmsnorm_pack_tail(const Tensor& input, const Tensor& weight, float epsilon, Tensor& output,
                       CudaExecutionView execution);

} // namespace ninfer::ops
