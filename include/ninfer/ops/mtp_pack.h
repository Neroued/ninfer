#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Op: mtp_pack_fc_input
 *
 * Math / indexing:
 *   out[0:D, t] = embedding_norm[:, t]
 *   out[D:2D, t] = hidden_norm[:, t]
 *
 * Logical shapes:
 *   BF16 embedding_norm and hidden_norm [D,T], out [2D,T], contiguous. The registered domains are
 *   D=5120 for Qwen3.6-27B and D=2048 for Qwen3.6-35B-A3B.
 *
 * Numeric:
 *   Exact BF16 element copies; no arithmetic or conversion.
 *
 * Effects:
 *   Writes the full output. Inputs and output must not alias.
 *
 * Workspace:
 *   None. The Op has no persistent state side effect.
 */
void mtp_pack_fc_input(const Tensor& embedding_norm, const Tensor& hidden_norm, Tensor& out,
                       cudaStream_t stream);

/**
 * Op: mtp_split_attn_in
 *
 * Math / indexing:
 *   For each token, rows [0,6144), [6144,7168), [7168,13312), and [13312,14336) are copied to
 *   flattened Q[6144], K[1024], Gate[6144], and V[1024], respectively.
 *
 * Logical shapes:
 *   attn_in [14336,T]; q/gate [256,24,T]; k/v [256,4,T], all contiguous BF16.
 *
 * Numeric:
 *   Exact BF16 element copies with only an index remap.
 *
 * Effects:
 *   Writes every output element. Outputs and input must be pairwise non-aliasing.
 *
 * Workspace:
 *   None. The Op has no persistent state side effect.
 */
void mtp_split_attn_in(const Tensor& attn_in, Tensor& q, Tensor& k, Tensor& gate, Tensor& v,
                       cudaStream_t stream);

} // namespace ninfer::ops
