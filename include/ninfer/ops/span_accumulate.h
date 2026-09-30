#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Accumulates columns [begin, begin + count) of a BF16 matrix into an FP32
 * column-sum vector, adding into the existing sum values (+=, not overwrite).
 *
 * x is contiguous BF16 [rows, tokens], addressed as x[row + rows * token];
 * sums is contiguous FP32 [rows]. count may be
 * zero (a no-op launch is permitted but must still be ordered on the stream). The
 * oracle evaluates the post-state sum as old_sum + sum of the represented BF16 columns
 * in FP64. The Op owns no allocation and leaves x unchanged. The working-set capture
 * path calls it once per attention layer over a chunk-local span of the pre-RoPE
 * query or key tensor.
 */
void span_accumulate(const Tensor& x, std::uint32_t begin, std::uint32_t count, Tensor& sums,
                     cudaStream_t stream);

} // namespace ninfer::ops
