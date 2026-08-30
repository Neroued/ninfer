#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Op: Target token log-probabilities
 *
 * Math / indexing:
 *   Let l[r,c] be the exact real value represented by logits[r,c]. For every column c,
 *
 *     ideal[c] = l[target_ids[c],c]
 *                - log(sum_{r=0..valid_rows-1} exp(l[r,c])).
 *
 * Logical shapes:
 *   logits is [physical_rows,C], target_ids is [C], and output is [C], with C>0 and
 *   1<=valid_rows<=physical_rows. Values in target_ids are in [0,valid_rows). Physical rows
 *   [valid_rows,physical_rows) do not participate in either the denominator or target lookup.
 *
 * Supported domain:
 *   logits is contiguous finite BF16, target_ids is contiguous I32, and output is contiguous
 *   FP32. Storage has its dtype's natural alignment.
 *
 * Numeric:
 *   output is the FP32 numerical approximation of ideal. Reduction association and private
 *   accumulator precision are implementation choices; the independent oracle evaluates the full
 *   formula in FP64 from the represented BF16 inputs.
 *
 * Effects:
 *   Writes every output element and preserves both inputs. Output must not overlap logits or
 *   target_ids.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Enqueues work on stream and owns no persistent state.
 */
void target_logprobs(const Tensor& logits, const Tensor& target_ids, std::int32_t valid_rows,
                     Tensor& output, cudaStream_t stream);

} // namespace ninfer::ops
