#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Prepare lane-owned BF16 prefixes as one compact ragged batch.
 *
 * source is BF16 [D,W,C], dense within each column with explicit column/lane strides.
 * lanes/starts/ends are I32 [B], destination is contiguous BF16 [D,W,B],
 * positions is contiguous I32 [W,B], and counts is I32 [B]. For N=ends[b]-starts[b], columns j<N
 * copy source[:,j,lanes[b]], use absolute position starts[b]+j, and publish counts[b]=N. The
 * physical tail j>=N is zero-filled and repeats the final live position (or starts[b] when N=0).
 * The caller guarantees 0<=N<=W and 0<=lanes[b]<C. D must be divisible by eight and all BF16
 * column starts must be 16-byte aligned. Physical W is independent of the live prefix N.
 * The three outputs are pairwise non-overlapping and do not overlap any input. Read-only inputs
 * may alias one another. Inputs remain unchanged; all
 * destination, position and count elements are overwritten exactly. The Op needs no transient
 * workspace and supports CUDA Graph capture/replay with changing device metadata.
 */
void prepare_ragged_prefix(const Tensor& source, const Tensor& lanes, const Tensor& starts,
                           const Tensor& ends, Tensor& destination, Tensor& positions,
                           Tensor& counts, cudaStream_t stream);

} // namespace ninfer::ops
