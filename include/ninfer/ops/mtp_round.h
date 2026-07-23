#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Op: mtp_prepare_alignment_ids
 *
 * Math / indexing:
 *   For T=K+1, alignment_ids[i]=verify_ids[i+1] for 0<=i<K, then
 *   alignment_ids[accepted]=token. alignment_ids[K] is written only when accepted=K; otherwise its
 *   previous value is retained.
 *
 * Logical shapes / effects:
 *   verify_ids and alignment_ids are distinct contiguous I32 [K+1], token and accepted are
 *   contiguous I32 scalars, 1<=K<=5, and 0<=accepted<=K. This prepares the shifted token inputs
 *   used to align the autoregressive MTP drafter after target verification. No workspace or other
 *   state is used.
 */
void mtp_prepare_alignment_ids(const Tensor& verify_ids, const Tensor& token,
                               const Tensor& accepted, Tensor& alignment_ids, cudaStream_t stream);

} // namespace ninfer::ops
