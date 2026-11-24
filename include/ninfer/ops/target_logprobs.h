#pragma once

#include "core/device.h"
#include "core/arena.h"
#include "core/tensor.h"

#include <cstdint>

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
 *   Enqueues work on execution.stream() and owns no persistent state.
 */
void target_logprobs(const Tensor& logits, const Tensor& target_ids, std::int32_t valid_rows,
                     Tensor& output, CudaExecutionView execution);

// Capacity for vocabulary_logprob_stats over 1..max_columns columns.
[[nodiscard]] std::size_t
vocabulary_logprob_stats_workspace_capacity_bytes(std::int32_t physical_rows,
                                                  std::int32_t max_columns);

/**
 * Local sufficient statistics for exact full-vocabulary target log-probabilities.
 * logits is finite BF16 [V,C]; local row v represents global ID token_begin+v, and only
 * IDs < token_domain participate. token_begin>=0, token_domain>0, V,C>0. target_ids is
 * I32 [C], each ID in [0,token_domain). Writes FP32 stats [3,C] containing:
 *   m = max(local logits), s = sum_local exp(logit-m),
 *   a = target logit if this partition owns the target, otherwise zero.
 * An empty partition writes (-infinity,0,0). Padding rows are not consumed.
 * Inputs/output are contiguous, naturally aligned Device views; output overlaps neither
 * input. Caller-owned workspace is disjoint and sized by the capacity query. No state changes.
 */
void vocabulary_logprob_stats(const Tensor& logits, const Tensor& target_ids,
                              std::int32_t token_begin, std::int32_t token_domain, Tensor& stats,
                              MemoryArena& workspace, CudaExecutionView execution);

/**
 * Given gathered FP32 stats [3,C,R] from R=1..16 disjoint partitions covering the token domain,
 * writes FP32 output [C]: sum_r a_r - M - log(sum_r s_r*exp(m_r-M)), M=max_r m_r.
 * Exactly one partition owns each target; empty partitions use (-infinity,0,0), and at least
 * one partition is nonempty. This computes the full-vocabulary formula of target_logprobs,
 * with the same FP32 numerical contract. Inputs/output are contiguous and naturally aligned
 * on execution.device(); output is disjoint from stats. No workspace, allocation or state.
 * Both entries support Graph capture/replay and consume current Device input values.
 */
void target_logprobs_from_stats(const Tensor& stats, Tensor& output, CudaExecutionView execution);

} // namespace ninfer::ops
