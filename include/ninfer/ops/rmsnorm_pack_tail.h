#pragma once

#include "core/device.h"
#include "core/tensor.h"

namespace ninfer::ops {

/**
 * output[d,(W-1)*b+i-1] = input[d,i,b] * weight[d]
 *                        / sqrt(sum_j input[j,i,b]^2 / D + 1e-6), for 1 <= i < W.
 *
 * Input is BF16 [D,W,B], weight is BF16 [D], and output is BF16 [D,(W-1)*B], with
 * D=5120, W=2..16 and B=1..8. All tensors are contiguous, non-null, 16-byte aligned and
 * pairwise disjoint. Uses plain multiplicative gain. Anchor columns input[:,0,b] are not read.
 * Writes all of output and preserves inputs. No workspace or state is used.
 */
void rmsnorm_pack_tail(const Tensor& input, const Tensor& weight, Tensor& output,
                       CudaExecutionView execution);

} // namespace ninfer::ops
