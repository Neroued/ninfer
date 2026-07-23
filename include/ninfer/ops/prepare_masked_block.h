#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Op: prepare one finite anchor-and-mask query block.
 *
 * anchor and length are distinct logical device I32 scalars. ids and positions are distinct,
 * contiguous I32 [B] outputs with:
 *
 *   ids[0]       = anchor[0]
 *   ids[i]       = mask_id,       1 <= i < B
 *   positions[i] = length[0] + i, 0 <= i < B
 *
 * The current registered domain is B=2..16 and nonnegative mask_id. The caller guarantees a
 * nonnegative length whose final emitted position is representable by I32. Inputs are unchanged,
 * every output element is overwritten, and the Op owns no workspace or persistent state.
 */
void prepare_masked_block(const Tensor& anchor, const Tensor& length, std::int32_t mask_id,
                          Tensor& ids, Tensor& positions, cudaStream_t stream);

} // namespace ninfer::ops
